"""
Synthetic World for Predictive Associative Memory Benchmark

Generates a world with rooms, objects, and trajectories through state space.
The key property: temporal co-occurrence creates associations that do NOT
align with embedding similarity. This lets us discriminate between
similarity-based retrieval and learned associative retrieval.

State embedding = room_centroid + Σ(object_embeddings) + action_embedding + noise
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import json
from pathlib import Path


@dataclass
class WorldConfig:
    """Configuration for synthetic world generation."""
    
    # Embedding space
    embedding_dim: int = 128
    
    # World structure
    num_rooms: int = 20
    num_objects: int = 50
    num_actions: int = 10
    
    # Room embeddings: moderate separation
    room_scale: float = 2.0
    
    # Object embeddings: comparable magnitude to rooms so objects matter for similarity
    object_scale: float = 1.5
    
    # Action embeddings: small modulation
    action_scale: float = 0.3
    
    # Noise
    noise_std: float = 0.3
    
    # Object-room affinities: probability of object appearing in each room
    # 0.0 = uniform (objects appear anywhere), 1.0 = strict (objects only in home rooms)
    room_affinity_strength: float = 0.5
    
    # Objects per room at any given time
    min_objects_per_state: int = 1
    max_objects_per_state: int = 4
    
    # Trajectory generation
    num_trajectories: int = 500
    trajectory_length: int = 100  # timesteps per trajectory
    room_persistence: float = 0.85  # probability of staying in same room next step
    object_persistence: float = 0.7  # probability each object stays next step
    
    # Temporal window for association ground truth
    temporal_window: int = 5  # states within this many steps are "co-occurring"
    
    # Random seed
    seed: int = 42


@dataclass
class State:
    """A single state in the world."""
    trajectory_id: int
    timestep: int
    room_id: int
    object_ids: List[int]
    action_id: int
    embedding: np.ndarray
    global_index: int = -1  # set after all states generated


@dataclass 
class Trajectory:
    """A sequence of states representing an episode of experience."""
    trajectory_id: int
    states: List[State]
    
    @property
    def length(self) -> int:
        return len(self.states)


class SyntheticWorld:
    """
    Generates a synthetic world with controlled associative structure.
    
    The world has rooms (clusters in embedding space), objects (features
    that can appear in any room), and trajectories (sequences of states
    representing lived experience).
    
    Key property: two states can be associated (temporal co-occurrence)
    without being similar (different rooms), and similar (same room)
    without being associated (no temporal proximity).
    """
    
    def __init__(self, config: WorldConfig = None):
        self.config = config or WorldConfig()
        self.rng = np.random.RandomState(self.config.seed)
        
        # Generate world structure
        self._generate_room_embeddings()
        self._generate_object_embeddings()
        self._generate_action_embeddings()
        self._generate_room_affinities()
        
        # Will be populated by generate_trajectories()
        self.trajectories: List[Trajectory] = []
        self.all_states: List[State] = []
        self.all_embeddings: np.ndarray = None  # [N, D] matrix
        
        # Ground truth (computed after trajectory generation)
        self._association_matrix: Optional[np.ndarray] = None
    
    def _generate_room_embeddings(self):
        """Generate well-separated room centroids."""
        d = self.config.embedding_dim
        n = self.config.num_rooms
        
        # Use random orthogonal vectors scaled for separation
        # QR decomposition of random matrix gives orthogonal columns
        if n <= d:
            random_matrix = self.rng.randn(d, n)
            Q, _ = np.linalg.qr(random_matrix)
            self.room_embeddings = Q[:, :n].T * self.config.room_scale  # [num_rooms, d]
        else:
            # More rooms than dimensions — use random unit vectors (less separated but ok)
            vecs = self.rng.randn(n, d)
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            self.room_embeddings = vecs * self.config.room_scale
    
    def _generate_object_embeddings(self):
        """Generate object embeddings at smaller scale."""
        d = self.config.embedding_dim
        n = self.config.num_objects
        
        # Random directions, smaller magnitude
        vecs = self.rng.randn(n, d)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        self.object_embeddings = vecs * self.config.object_scale  # [num_objects, d]
    
    def _generate_action_embeddings(self):
        """Generate action embeddings as small modulations."""
        d = self.config.embedding_dim
        n = self.config.num_actions
        
        vecs = self.rng.randn(n, d)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        self.action_embeddings = vecs * self.config.action_scale  # [num_actions, d]
    
    def _generate_room_affinities(self):
        """
        Generate object-room affinity matrix.
        
        Each object has a "home room" where it's most likely to appear,
        but can appear in any room with some probability.
        """
        n_obj = self.config.num_objects
        n_rooms = self.config.num_rooms
        strength = self.config.room_affinity_strength
        
        # Assign each object a home room
        self.object_home_rooms = self.rng.randint(0, n_rooms, size=n_obj)
        
        # Build affinity matrix: [num_objects, num_rooms]
        # Base probability is uniform
        base_prob = np.ones((n_obj, n_rooms)) / n_rooms
        
        # Add bias toward home room
        home_bias = np.zeros((n_obj, n_rooms))
        for obj_id in range(n_obj):
            home_bias[obj_id, self.object_home_rooms[obj_id]] = strength
        
        # Combine and normalize
        self.room_affinities = base_prob + home_bias
        self.room_affinities /= self.room_affinities.sum(axis=1, keepdims=True)
    
    def _sample_objects_for_room(self, room_id: int) -> List[int]:
        """Sample which objects are present in a room state."""
        n_objects = self.rng.randint(
            self.config.min_objects_per_state, 
            self.config.max_objects_per_state + 1
        )
        
        # Weight by affinity to this room
        probs = self.room_affinities[:, room_id]
        probs = probs / probs.sum()
        
        chosen = self.rng.choice(
            self.config.num_objects, 
            size=min(n_objects, self.config.num_objects),
            replace=False, 
            p=probs
        )
        return sorted(chosen.tolist())
    
    def _compute_state_embedding(self, room_id: int, object_ids: List[int], 
                                  action_id: int) -> np.ndarray:
        """Compute the embedding for a state."""
        emb = self.room_embeddings[room_id].copy()
        
        for obj_id in object_ids:
            emb += self.object_embeddings[obj_id]
        
        emb += self.action_embeddings[action_id]
        emb += self.rng.randn(self.config.embedding_dim) * self.config.noise_std
        
        return emb
    
    def generate_trajectories(self) -> List[Trajectory]:
        """
        Generate all trajectories through the world.
        
        Each trajectory represents an episode of experience: the agent
        moves through rooms, interacts with objects, performs actions.
        """
        self.trajectories = []
        self.all_states = []
        global_idx = 0
        
        for traj_id in range(self.config.num_trajectories):
            states = []
            
            # Start in a random room with random objects
            current_room = self.rng.randint(0, self.config.num_rooms)
            current_objects = self._sample_objects_for_room(current_room)
            
            for t in range(self.config.trajectory_length):
                # Maybe change room
                if self.rng.random() > self.config.room_persistence:
                    current_room = self.rng.randint(0, self.config.num_rooms)
                    # Room change = mostly new objects
                    current_objects = self._sample_objects_for_room(current_room)
                else:
                    # Maybe swap some objects (persistence within room)
                    new_objects = []
                    for obj_id in current_objects:
                        if self.rng.random() < self.config.object_persistence:
                            new_objects.append(obj_id)
                    
                    # Maybe add new objects
                    if len(new_objects) < self.config.min_objects_per_state or \
                       (len(new_objects) < self.config.max_objects_per_state and 
                        self.rng.random() < 0.3):
                        extra = self._sample_objects_for_room(current_room)
                        for obj_id in extra:
                            if obj_id not in new_objects and \
                               len(new_objects) < self.config.max_objects_per_state:
                                new_objects.append(obj_id)
                    
                    current_objects = sorted(new_objects) if new_objects else \
                                     self._sample_objects_for_room(current_room)
                
                # Random action
                action = self.rng.randint(0, self.config.num_actions)
                
                # Compute embedding
                embedding = self._compute_state_embedding(
                    current_room, current_objects, action
                )
                
                state = State(
                    trajectory_id=traj_id,
                    timestep=t,
                    room_id=current_room,
                    object_ids=current_objects.copy(),
                    action_id=action,
                    embedding=embedding,
                    global_index=global_idx
                )
                states.append(state)
                self.all_states.append(state)
                global_idx += 1
            
            self.trajectories.append(Trajectory(traj_id, states))
        
        # Build embedding matrix
        self.all_embeddings = np.stack([s.embedding for s in self.all_states])
        
        print(f"Generated {len(self.trajectories)} trajectories, "
              f"{len(self.all_states)} total states")
        print(f"Embedding matrix shape: {self.all_embeddings.shape}")
        
        return self.trajectories
    
    def compute_association_ground_truth(self, cross_trajectory=False,
                                         cross_traj_strength=0.3) -> np.ndarray:
        """
        Compute ground truth association matrix based on temporal co-occurrence.

        States within temporal_window steps of each other in the same trajectory
        are associated. Strength = number of co-occurrences across all trajectories
        weighted by temporal proximity (closer = stronger).

        If cross_trajectory=True, also adds associations between states in
        DIFFERENT trajectories that share objects. This models an agent that
        recognizes previously-seen objects in new contexts. The cross-trajectory
        strength is lower than temporal co-occurrence (recognition vs co-presence).

        Returns sparse-ish matrix since most pairs aren't associated.
        We store as dict for memory efficiency.
        """
        if not self.all_states:
            raise ValueError("Generate trajectories first")

        N = len(self.all_states)
        window = self.config.temporal_window

        # Build association dict: (i, j) -> strength
        # Using dict instead of dense matrix for memory efficiency
        associations = {}

        for traj in self.trajectories:
            n_states = len(traj.states)
            for i in range(n_states):
                si = traj.states[i]
                for j in range(i + 1, min(i + window + 1, n_states)):
                    sj = traj.states[j]

                    # Temporal proximity weighting: closer = stronger
                    dt = j - i
                    weight = 1.0 / dt  # inverse temporal distance

                    gi, gj = si.global_index, sj.global_index
                    key = (min(gi, gj), max(gi, gj))
                    associations[key] = associations.get(key, 0.0) + weight

        temporal_count = len(associations)

        # Cross-trajectory object-mediated associations
        if cross_trajectory:
            # Build object -> list of (state_index, trajectory_id) map
            obj_states = {}
            for s in self.all_states:
                for obj_id in s.object_ids:
                    obj_states.setdefault(obj_id, []).append(
                        (s.global_index, s.trajectory_id))

            cross_count = 0
            for obj_id, state_list in obj_states.items():
                # Group by trajectory
                by_traj = {}
                for gi, tid in state_list:
                    by_traj.setdefault(tid, []).append(gi)

                traj_ids = list(by_traj.keys())
                if len(traj_ids) < 2:
                    continue

                # For each pair of trajectories, associate states sharing this object
                # Subsample to avoid quadratic explosion
                for ti in range(len(traj_ids)):
                    for tj in range(ti + 1, len(traj_ids)):
                        t1_states = by_traj[traj_ids[ti]]
                        t2_states = by_traj[traj_ids[tj]]

                        # Subsample: max 3 states per trajectory per object pair
                        if len(t1_states) > 3:
                            t1_sample = self.rng.choice(t1_states, 3, replace=False).tolist()
                        else:
                            t1_sample = t1_states
                        if len(t2_states) > 3:
                            t2_sample = self.rng.choice(t2_states, 3, replace=False).tolist()
                        else:
                            t2_sample = t2_states

                        for gi in t1_sample:
                            for gj in t2_sample:
                                key = (min(gi, gj), max(gi, gj))
                                # Add cross-trajectory strength, scaled by number
                                # of shared objects (more shared = stronger)
                                associations[key] = associations.get(key, 0.0) + cross_traj_strength
                                cross_count += 1

            print(f"Cross-trajectory associations added: {cross_count} links")

        self._associations = associations

        print(f"Ground truth associations: {len(associations)} pairs "
              f"(temporal: {temporal_count}, total: {len(associations)}) "
              f"out of {N*(N-1)//2} possible ({100*len(associations)/(N*(N-1)//2):.2f}%)")

        return associations
    
    def get_association_strength(self, idx_i: int, idx_j: int) -> float:
        """Get association strength between two states."""
        if self._associations is None:
            raise ValueError("Compute ground truth first")
        key = (min(idx_i, idx_j), max(idx_i, idx_j))
        return self._associations.get(key, 0.0)

    def split_associations(self, train_ratio: float = 0.7, seed: int = 42) -> Tuple[Dict, Dict]:
        """
        Split associations into edge-disjoint train and test sets.

        Returns:
            (train_associations, test_associations) -- both dicts of (i,j)->strength
        """
        if self._associations is None:
            raise ValueError("Compute ground truth first")

        rng = np.random.RandomState(seed)
        keys = list(self._associations.keys())
        rng.shuffle(keys)

        split_idx = int(len(keys) * train_ratio)
        train_keys = set(keys[:split_idx])

        train_assoc = {k: self._associations[k] for k in keys[:split_idx]}
        test_assoc = {k: self._associations[k] for k in keys[split_idx:]}

        print(f"Association split: {len(train_assoc)} train ({100*train_ratio:.0f}%), "
              f"{len(test_assoc)} test ({100*(1-train_ratio):.0f}%)")

        return train_assoc, test_assoc
    
    def build_test_sets(self) -> Dict[str, List[Tuple]]:
        """
        Build the four test sets that discriminate associative from similarity retrieval.
        
        Returns dict with:
          - 'high_sim_low_assoc': pairs that are similar but NOT associated
          - 'low_sim_high_assoc': pairs that are dissimilar but ARE associated  
          - 'transitive_2hop': (a, b, c) where a-b and b-c are associated but a-c are not directly
          - 'creative_bridge': cross-trajectory transitive chains
        """
        if self._associations is None:
            self.compute_association_ground_truth()
        
        N = len(self.all_states)
        embeddings = self.all_embeddings
        
        # Compute pairwise similarities for a sample (full matrix too large)
        sample_size = min(5000, N)
        sample_indices = self.rng.choice(N, size=sample_size, replace=False)
        
        test_sets = {
            'high_sim_low_assoc': [],
            'low_sim_high_assoc': [],
            'transitive_2hop': [],
            'creative_bridge': []
        }
        
        # --- Test 1: High similarity, low association ---
        print("Building high-similarity low-association pairs...")
        for idx in sample_indices[:1000]:
            # Find states in the same room but different trajectories
            state = self.all_states[idx]
            candidates = [
                s for s in self.all_states 
                if s.room_id == state.room_id 
                and s.trajectory_id != state.trajectory_id
            ]
            if candidates:
                other = self.rng.choice(candidates)
                assoc = self.get_association_strength(idx, other.global_index)
                if assoc == 0.0:  # no association
                    sim = np.dot(embeddings[idx], embeddings[other.global_index]) / \
                          (np.linalg.norm(embeddings[idx]) * np.linalg.norm(embeddings[other.global_index]))
                    if sim > 0.5:  # reasonably similar
                        test_sets['high_sim_low_assoc'].append(
                            (idx, other.global_index, float(sim), float(assoc))
                        )
        
        # --- Test 2: Low similarity, high association ---
        print("Building low-similarity high-association pairs...")
        # Find associated pairs where room changed (object bridge)
        for (i, j), strength in list(self._associations.items())[:50000]:
            si, sj = self.all_states[i], self.all_states[j]
            if si.room_id != sj.room_id:  # different rooms
                sim = np.dot(embeddings[i], embeddings[j]) / \
                      (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                if sim < 0.3:  # low similarity
                    # Check for shared objects (the bridge)
                    shared = set(si.object_ids) & set(sj.object_ids)
                    test_sets['low_sim_high_assoc'].append(
                        (i, j, float(sim), float(strength), list(shared))
                    )
                    if len(test_sets['low_sim_high_assoc']) >= 1000:
                        break
        
        # --- Test 3: Transitive 2-hop associations ---
        print("Building transitive 2-hop test cases...")
        # Find chains: a associated with b, b associated with c, a not associated with c
        assoc_by_state = {}
        for (i, j), strength in self._associations.items():
            if strength > 0.5:  # only strong associations
                assoc_by_state.setdefault(i, set()).add(j)
                assoc_by_state.setdefault(j, set()).add(i)
        
        checked = 0
        for a in self.rng.choice(list(assoc_by_state.keys()), 
                                  size=min(2000, len(assoc_by_state)), replace=False):
            if checked > 10000:
                break
            neighbors_a = assoc_by_state.get(a, set())
            for b in list(neighbors_a)[:10]:
                neighbors_b = assoc_by_state.get(b, set())
                for c in list(neighbors_b)[:10]:
                    checked += 1
                    if c != a and c not in neighbors_a:
                        # a-b associated, b-c associated, a-c NOT directly associated
                        test_sets['transitive_2hop'].append((a, b, c))
                        if len(test_sets['transitive_2hop']) >= 1000:
                            break
                if len(test_sets['transitive_2hop']) >= 1000:
                    break
            if len(test_sets['transitive_2hop']) >= 1000:
                break
        
        # --- Test 4: Creative bridges (cross-trajectory transitive chains) ---
        print("Building creative bridge test cases...")
        # Find object-mediated cross-trajectory links
        # Object X appears in trajectory T1 and trajectory T2
        # States near object X in T1 should eventually reach states near object X in T2
        obj_to_trajectories = {}
        for state in self.all_states:
            for obj_id in state.object_ids:
                key = (obj_id, state.trajectory_id)
                if key not in obj_to_trajectories:
                    obj_to_trajectories[key] = []
                obj_to_trajectories[key].append(state.global_index)
        
        # Find objects that appear in multiple trajectories
        obj_traj_map = {}
        for (obj_id, traj_id), state_indices in obj_to_trajectories.items():
            obj_traj_map.setdefault(obj_id, {})[traj_id] = state_indices
        
        for obj_id, traj_dict in obj_traj_map.items():
            if len(traj_dict) < 2:
                continue
            traj_ids = list(traj_dict.keys())
            for t1_idx in range(len(traj_ids)):
                for t2_idx in range(t1_idx + 1, len(traj_ids)):
                    t1, t2 = traj_ids[t1_idx], traj_ids[t2_idx]
                    # Pick a state from each trajectory that contains this object
                    s1 = self.rng.choice(traj_dict[t1])
                    s2 = self.rng.choice(traj_dict[t2])
                    
                    # Verify they're not directly associated
                    direct = self.get_association_strength(s1, s2)
                    if direct == 0.0:
                        test_sets['creative_bridge'].append(
                            (s1, s2, obj_id, t1, t2)
                        )
                        if len(test_sets['creative_bridge']) >= 1000:
                            break
                if len(test_sets['creative_bridge']) >= 1000:
                    break
            if len(test_sets['creative_bridge']) >= 1000:
                break
        
        for name, pairs in test_sets.items():
            print(f"  {name}: {len(pairs)} test cases")
        
        return test_sets
    
    def get_training_pairs(self, num_negatives: int = 15, max_pairs: int = 100000,
                           associations: Dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate training data: (anchor, positive, negatives) triplets.

        Uses hard negative mining: half the negatives come from the same room
        as the positive (hard negatives that are similar but not associated),
        the other half are random (easy negatives from different rooms).

        Args:
            associations: If provided, use only these associations (e.g. train split).
                          If None, uses all associations (legacy behavior).
        """
        if self._associations is None:
            self.compute_association_ground_truth()

        source_assoc = associations if associations is not None else self._associations

        # Build room-to-state index for hard negative sampling
        room_to_states = {}
        for s in self.all_states:
            room_to_states.setdefault(s.room_id, []).append(s.global_index)
        for room_id in room_to_states:
            room_to_states[room_id] = np.array(room_to_states[room_id])

        # Build per-state association set for fast lookup
        assoc_set = set()
        for (i, j) in source_assoc:
            assoc_set.add((i, j))
            assoc_set.add((j, i))

        assoc_list = [(i, j, s) for (i, j), s in source_assoc.items() if s >= 0.2]
        self.rng.shuffle(assoc_list)

        # Use both directions (i->j and j->i) to double effective data
        augmented = []
        for i, j, s in assoc_list:
            augmented.append((i, j, s))
            augmented.append((j, i, s))
        self.rng.shuffle(augmented)
        augmented = augmented[:max_pairs]

        N = len(self.all_states)
        num_hard = num_negatives // 2
        num_easy = num_negatives - num_hard

        anchors = []
        positives = []
        negatives_list = []

        for i, j, strength in augmented:
            anchors.append(self.all_embeddings[i])
            positives.append(self.all_embeddings[j])

            neg_indices = []

            # Hard negatives: same room as positive, not associated with anchor
            pos_room = self.all_states[j].room_id
            room_states = room_to_states[pos_room]
            candidates = room_states[room_states != i]  # exclude anchor
            candidates = candidates[candidates != j]     # exclude positive
            if len(candidates) >= num_hard:
                chosen = self.rng.choice(candidates, size=num_hard, replace=False)
                neg_indices.extend(chosen.tolist())
            else:
                neg_indices.extend(candidates.tolist())

            # Fill remaining with easy random negatives
            remaining = num_negatives - len(neg_indices)
            if remaining > 0:
                random_neg = self.rng.choice(N, size=remaining * 2, replace=False)
                # Filter out anchor, positive, and already-chosen negatives
                exclude = {i, j} | set(neg_indices)
                random_neg = [n for n in random_neg if n not in exclude][:remaining]
                neg_indices.extend(random_neg)

            # Pad if needed
            while len(neg_indices) < num_negatives:
                neg_indices.append(self.rng.randint(N))

            negatives_list.append(self.all_embeddings[neg_indices[:num_negatives]])

        anchors = np.stack(anchors)
        positives = np.stack(positives)
        negatives = np.stack(negatives_list)

        print(f"Generated {len(anchors)} training triplets")
        print(f"  Anchors: {anchors.shape}")
        print(f"  Positives: {positives.shape}")
        print(f"  Negatives: {negatives.shape}")
        print(f"  Hard negatives per sample: {num_hard}, Easy: {num_easy}")

        return anchors, positives, negatives
    
    def summary(self) -> str:
        """Print world summary statistics."""
        lines = [
            "=" * 60,
            "Synthetic World Summary",
            "=" * 60,
            f"Embedding dim: {self.config.embedding_dim}",
            f"Rooms: {self.config.num_rooms}",
            f"Objects: {self.config.num_objects}",
            f"Actions: {self.config.num_actions}",
            f"Trajectories: {len(self.trajectories)}",
            f"Total states: {len(self.all_states)}",
            f"Temporal window: {self.config.temporal_window}",
        ]
        
        if self._associations:
            lines.append(f"Association pairs: {len(self._associations)}")
        
        # Room visit distribution
        room_counts = {}
        for s in self.all_states:
            room_counts[s.room_id] = room_counts.get(s.room_id, 0) + 1
        lines.append(f"Room visit range: {min(room_counts.values())}-{max(room_counts.values())}")
        
        # Object occurrence distribution
        obj_counts = {}
        for s in self.all_states:
            for obj_id in s.object_ids:
                obj_counts[obj_id] = obj_counts.get(obj_id, 0) + 1
        lines.append(f"Object occurrence range: {min(obj_counts.values())}-{max(obj_counts.values())}")
        
        # Cross-room object bridges
        obj_rooms = {}
        for s in self.all_states:
            for obj_id in s.object_ids:
                obj_rooms.setdefault(obj_id, set()).add(s.room_id)
        multi_room_objs = sum(1 for rooms in obj_rooms.values() if len(rooms) > 1)
        lines.append(f"Objects appearing in multiple rooms: {multi_room_objs}/{self.config.num_objects}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    config = WorldConfig(
        num_trajectories=50,
        trajectory_length=20,
    )
    world = SyntheticWorld(config)
    world.generate_trajectories()
    world.compute_association_ground_truth()
    print(world.summary())
    
    test_sets = world.build_test_sets()
    
    # Quick similarity check
    emb = world.all_embeddings
    # Same room pairs vs different room pairs
    same_room_sims = []
    diff_room_sims = []
    for _ in range(1000):
        i, j = np.random.choice(len(world.all_states), 2, replace=False)
        sim = np.dot(emb[i], emb[j]) / (np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]))
        if world.all_states[i].room_id == world.all_states[j].room_id:
            same_room_sims.append(sim)
        else:
            diff_room_sims.append(sim)
    
    print(f"\nSimilarity sanity check:")
    print(f"  Same room: {np.mean(same_room_sims):.3f} ± {np.std(same_room_sims):.3f}")
    print(f"  Diff room: {np.mean(diff_room_sims):.3f} ± {np.std(diff_room_sims):.3f}")
