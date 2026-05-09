"""
GoT-HCS: Graph-of-Thoughts Enhanced Hierarchical Clinical Summarization

Transforms lengthy clinical notes into Brief Hospital Course (BHC) summaries through graph-based reasoning and medical knowledge integration.

Pipeline:
1. Graph Construction: Extracts temporal clinical entities as nodes (ClinicalThought) 
   with UMLS linking, builds weighted edges (temporal/causal/logical) via GraphConstructor.

2. Graph Encoding: 3-layer GAT enriches node embeddings through neighbor aggregation.

3. Knowledge Augmentation: Fuses UMLS biomedical knowledge into nodes via attention-based 
   BiomedicalKnowledgeGraph retrieval.

4. Multi-Stage GoT Reasoning: Iteratively generates multiple summary candidates, aggregates 
   cross-node information, and refines based on relevance/consistency scoring (2-3 iterations).

5. Hierarchical Distillation: Clusters thoughts into themes, generates section summaries, 
   distills into final compressed representation.

6. Text Generation: T5-based fluent BHC generation with beam search.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass, field
import re
import json
from pathlib import Path


@dataclass
class ClinicalThought:
    """Represents a single clinical reasoning unit (graph node)"""
    text: str
    category: str  # 'diagnostic', 'therapeutic', 'monitoring', 'outcome'
    entity_type: str  # 'diagnosis', 'medication', 'procedure', 'lab_result'
    embedding: Optional[torch.Tensor] = None  
    timestamp: Optional[float] = None  
    score: float = 0.0
    linked_entities: List[str] = field(default_factory=list)  # UMLS CUIs
    
    
@dataclass
class ThoughtEdge:
    """Represents dependency between clinical thoughts"""
    source_idx: int
    target_idx: int
    edge_type: str  # 'temporal', 'causal', 'logical'
    weight: float = 1.0


class BiomedicalKnowledgeGraph:
    """
    Loads and manages biomedical knowledge graph from UMLS/SNOMED-CT
    """
    def __init__(self, 
                 umls_embeddings_path: Optional[str] = None,
                 umls_relations_path: Optional[str] = None,
                 embedding_dim: int = 768):
        """
        Args:
            umls_embeddings_path: Path to pre-trained UMLS concept embeddings (e.g., from SapBERT, CODER, BioLORD)
            umls_relations_path: Path to UMLS relationship triples
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.cui_to_idx = {}
        self.idx_to_cui = {}
        self.cui_embeddings = None
        self.relations = {}  # {(cui1, cui2): relation_type}
        
        # Load embeddings if path provided
        if umls_embeddings_path and Path(umls_embeddings_path).exists():
            self._load_umls_embeddings(umls_embeddings_path)
        else:
            print(f"Warning: UMLS embeddings not found at {umls_embeddings_path}. Using zero initialization.")
            self._initialize_default_embeddings()
        
        # Load relations if path provided
        if umls_relations_path and Path(umls_relations_path).exists():
            self._load_umls_relations(umls_relations_path)
        else:
            print(f"Warning: UMLS relations not found at {umls_relations_path}. Skipping relation loading.")
    
    def _initialize_default_embeddings(self):
        """Initialize with common medical concepts (fallback)"""
        # Common medical CUIs (this is a minimal fallback - in practice, load full UMLS)
        common_cuis = [
            'C0030193',  # Pain
            'C0015967',  # Fever
            'C0010200',  # Cough
            'C0013404',  # Dyspnea
            'C0020538',  # Hypertension
            'C0011847',  # Diabetes
            'C0003232',  # Antibiotic
            'C0030705',  # Patient
            'C0184666',  # Hospital admission
            'C0030685',  # Patient discharge
        ]
        
        self.cui_to_idx = {cui: idx for idx, cui in enumerate(common_cuis)}
        self.idx_to_cui = {idx: cui for idx, cui in enumerate(common_cuis)}
        
        # Initialize with small random embeddings
        num_concepts = len(common_cuis)
        self.cui_embeddings = torch.randn(num_concepts, self.embedding_dim) * 0.01
    
    def _load_umls_embeddings(self, embeddings_path: str):
        """
        Load pre-trained UMLS embeddings
        Expected format: JSON/NPZ with {CUI: embedding_vector}
        """
        path = Path(embeddings_path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.cui_to_idx = {cui: idx for idx, cui in enumerate(data.keys())}
            self.idx_to_cui = {idx: cui for cui, idx in self.cui_to_idx.items()}
            
            embeddings_list = [data[cui] for cui in self.idx_to_cui.values()]
            self.cui_embeddings = torch.tensor(embeddings_list, dtype=torch.float32)
        
        elif path.suffix == '.npz':
            data = np.load(path, allow_pickle=True)
            self.cui_to_idx = data['cui_to_idx'].item()
            self.idx_to_cui = {v: k for k, v in self.cui_to_idx.items()}
            self.cui_embeddings = torch.tensor(data['embeddings'], dtype=torch.float32)
        
        print(f"Loaded {len(self.cui_to_idx)} UMLS concept embeddings from {embeddings_path}")
    
    def _load_umls_relations(self, relations_path: str):
        """
        Load UMLS semantic relations
        Expected format: TSV with columns [CUI1, RELATION_TYPE, CUI2]
        """
        with open(relations_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    cui1, rel_type, cui2 = parts[0], parts[1], parts[2]
                    self.relations[(cui1, cui2)] = rel_type
        
        print(f"Loaded {len(self.relations)} UMLS relations from {relations_path}")
    
    def get_embedding(self, cui: str) -> Optional[torch.Tensor]:
        """Get embedding for a specific CUI"""
        idx = self.cui_to_idx.get(cui)
        if idx is not None and self.cui_embeddings is not None:
            return self.cui_embeddings[idx]
        return None
    
    def get_embeddings_batch(self, cuis: List[str]) -> torch.Tensor:
        """Get embeddings for multiple CUIs"""
        embeddings = []
        for cui in cuis:
            emb = self.get_embedding(cui)
            if emb is not None:
                embeddings.append(emb)
        
        if len(embeddings) == 0:
            # Return zero embedding if no CUIs found
            return torch.zeros(1, self.embedding_dim)
        
        return torch.stack(embeddings)
    
    def get_related_concepts(self, cui: str, max_hops: int = 2) -> List[str]:
        """Get related concepts through UMLS relations"""
        if not self.relations:
            return []
        
        related = set()
        current_level = {cui}
        
        for _ in range(max_hops):
            next_level = set()
            for c in current_level:
                # Find all relations where c is source or target
                for (cui1, cui2), rel_type in self.relations.items():
                    if cui1 == c:
                        next_level.add(cui2)
                    elif cui2 == c:
                        next_level.add(cui1)
            
            related.update(next_level)
            current_level = next_level
            
            if len(related) > 100:  # Limit expansion
                break
        
        return list(related)[:50]  # Return top 50


class ClinicalEntityLinker(nn.Module):
    """
    Links clinical text mentions to UMLS/SNOMED-CT concepts
    """
    def __init__(self, 
                 biobert_model: str = "dmis-lab/biobert-base-cased-v1.2",
                 kg: BiomedicalKnowledgeGraph = None):
        super().__init__()
        
        self.biobert = AutoModel.from_pretrained(biobert_model)
        self.tokenizer = AutoTokenizer.from_pretrained(biobert_model)
        self.kg = kg
        
        # Entity linker scoring
        self.similarity_scorer = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 768)
        )
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def link_entity(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Link text mention to UMLS CUIs
        
        Returns:
            List of (CUI, confidence_score) tuples
        """
        if self.kg is None or self.kg.cui_embeddings is None:
            return []
        
        device = self.device
        
        # Encode text mention
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.biobert(**inputs)
            mention_embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
        
        # Project mention embedding
        mention_proj = self.similarity_scorer(mention_embedding)  # [1, 768]
        
        # Compute similarity with all CUIs
        cui_embeddings = self.kg.cui_embeddings.to(device)  # [num_cuis, 768]
        
        # Cosine similarity
        mention_norm = F.normalize(mention_proj, p=2, dim=-1)
        cui_norm = F.normalize(cui_embeddings, p=2, dim=-1)
        
        similarities = torch.matmul(mention_norm, cui_norm.T).squeeze(0)  # [num_cuis]
        
        # Get top-k
        top_scores, top_indices = similarities.topk(min(top_k, len(similarities)))
        
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            cui = self.kg.idx_to_cui.get(idx, "UNKNOWN")
            results.append((cui, score))
        
        return results


class TemporalClinicalEntityExtractor(nn.Module):
    """
    Module 1: Clinical Thought Graph Construction
    Extracts temporal anchors and clinical entities from input text
    """
    def __init__(self, 
                 model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 kg: Optional[BiomedicalKnowledgeGraph] = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Knowledge graph
        self.kg = kg
        
        # Entity linker
        if kg is not None:
            self.entity_linker = ClinicalEntityLinker(kg=kg)
        else:
            self.entity_linker = None
        
        # Entity type classifier
        self.entity_classifier = nn.Linear(768, 5)  # diagnosis, medication, procedure, lab, other
        
        # Temporal anchor detection
        self.temporal_detector = nn.Linear(768, 1)
        
        # Category classifier
        self.category_classifier = nn.Linear(768, 4)  # diagnostic, therapeutic, monitoring, outcome
        
        # Temporal patterns
        self.temporal_patterns = [
            r'on\s+admission', r'hospital\s+day\s+\d+', r'post-operative\s+day\s+\d+',
            r'on\s+discharge', r'\d{1,2}/\d{1,2}/\d{2,4}', r'day\s+\d+'
        ]
    
    @property
    def device(self):
        return next(self.parameters()).device
        
    def extract_entities(self, text: str) -> List[ClinicalThought]:
        """Extract clinical entities and create thought nodes"""
        device = self.device
        
        # Tokenize and encode
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, 
                               truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        thoughts = []
        
        for idx, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:
                continue
                
            # Get sentence embedding
            sent_inputs = self.tokenizer(sentence, return_tensors='pt', 
                                        padding=True, truncation=True)
            sent_inputs = {k: v.to(device) for k, v in sent_inputs.items()}
            
            with torch.no_grad():
                sent_outputs = self.encoder(**sent_inputs)
                sent_embedding = sent_outputs.last_hidden_state.mean(dim=1)
            
            # Classify entity type
            entity_logits = self.entity_classifier(sent_embedding)
            entity_type = ['diagnosis', 'medication', 'procedure', 'lab_result', 'other'][
                entity_logits.argmax(dim=-1).item()
            ]
            
            # Classify category
            category_logits = self.category_classifier(sent_embedding)
            category = ['diagnostic', 'therapeutic', 'monitoring', 'outcome'][
                category_logits.argmax(dim=-1).item()
            ]
            
            # Extract temporal information
            timestamp = self._extract_timestamp(sentence)
            
            # Link to UMLS concepts
            linked_cuis = []
            if self.entity_linker is not None:
                cui_matches = self.entity_linker.link_entity(sentence, top_k=3)
                linked_cuis = [cui for cui, score in cui_matches if score > 0.5]
            
            thought = ClinicalThought(
                text=sentence.strip(),
                category=category,
                entity_type=entity_type,
                embedding=sent_embedding.squeeze(0).detach(),
                timestamp=timestamp,
                linked_entities=linked_cuis
            )
            thoughts.append(thought)
        
        return thoughts
    
    def _extract_timestamp(self, text: str) -> Optional[float]:
        """Extract temporal anchor from text"""
        for pattern in self.temporal_patterns:
            match = re.search(pattern, text.lower())
            if match:
                day_match = re.search(r'\d+', match.group())
                if day_match:
                    return float(day_match.group())
        return None


class GraphConstructor(nn.Module):
    """
    Module 1 (continued): Constructs dependency edges between thoughts
    """
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        # Edge type predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)  # temporal, causal, logical
        )
        
        # Edge weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    @property
    def device(self):
        return next(self.parameters()).device
        
    def construct_edges(self, thoughts: List[ClinicalThought]) -> List[ThoughtEdge]:
        """Construct edges between thought nodes"""
        edges = []
        device = self.device
        
        for i in range(len(thoughts)):
            for j in range(i + 1, len(thoughts)):
                # Concatenate embeddings
                emb_i = thoughts[i].embedding.to(device)
                emb_j = thoughts[j].embedding.to(device)
                edge_input = torch.cat([emb_i, emb_j], dim=-1)
                
                with torch.no_grad():
                    # Predict edge type
                    edge_logits = self.edge_predictor(edge_input.unsqueeze(0))
                    edge_type_idx = edge_logits.argmax(dim=-1).item()
                    edge_type = ['temporal', 'causal', 'logical'][edge_type_idx]
                    
                    # Predict edge weight
                    weight = self.weight_predictor(edge_input.unsqueeze(0)).item()
                
                # Add edge if weight is significant
                if weight > 0.3:
                    # Check temporal consistency
                    if thoughts[i].timestamp is not None and thoughts[j].timestamp is not None:
                        if thoughts[i].timestamp <= thoughts[j].timestamp:
                            edges.append(ThoughtEdge(i, j, edge_type, weight))
                    else:
                        edges.append(ThoughtEdge(i, j, edge_type, weight))
        
        return edges


class ThoughtGraphEncoder(nn.Module):
    """
    Graph neural network to encode thought graph structure
    """
    def __init__(self, hidden_dim: int = 768, num_layers: int = 3):
        super().__init__()
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=4, concat=False, add_self_loops=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, hidden_dim]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
        """
        x = node_features
        
        # Handle empty edge case
        if edge_index.numel() == 0:
            for norm in self.norms:
                x = norm(x)
                x = F.relu(x)
            return x
        
        for conv, norm in zip(self.convs, self.norms):
            if edge_weight is not None and edge_weight.numel() > 0:
                edge_attr = edge_weight.unsqueeze(-1)
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        return x


class KnowledgeGraphAugmentation(nn.Module):
    """
    Module 3: Knowledge-Enhanced Context Integration
    Integrates real biomedical KG from UMLS/SNOMED-CT
    """
    def __init__(self, 
                 kg: BiomedicalKnowledgeGraph,
                 kg_embedding_dim: int = 768):
        super().__init__()
        self.kg = kg
        self.kg_embedding_dim = kg_embedding_dim
        
        # Knowledge integration via attention
        self.knowledge_fusion = nn.MultiheadAttention(
            embed_dim=kg_embedding_dim, num_heads=8, batch_first=True
        )
        
        # Knowledge context encoder
        self.kg_context_encoder = nn.Sequential(
            nn.Linear(kg_embedding_dim, kg_embedding_dim),
            nn.LayerNorm(kg_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(kg_embedding_dim * 2, kg_embedding_dim),
            nn.Sigmoid()
        )
    
    def retrieve_knowledge(self, thoughts: List[ClinicalThought], top_k: int = 5) -> List[torch.Tensor]:
        """
        Retrieve relevant knowledge from biomedical KG for each thought
        
        Returns:
            List of knowledge embeddings, one tensor per thought
        """
        knowledge_embeddings = []
        
        for thought in thoughts:
            # Get linked CUIs for this thought
            cuis = thought.linked_entities
            
            if len(cuis) == 0:
                # No linked entities - return zero embedding
                knowledge_embeddings.append(
                    torch.zeros(1, self.kg_embedding_dim)
                )
                continue
            
            # Get embeddings for linked CUIs
            cui_embeds = self.kg.get_embeddings_batch(cuis)  # [num_cuis, embed_dim]
            
            # Also get related concepts
            related_cuis = []
            for cui in cuis[:3]:  # Limit to top 3 to avoid explosion
                related = self.kg.get_related_concepts(cui, max_hops=1)
                related_cuis.extend(related[:5])  # Top 5 related per CUI
            
            if len(related_cuis) > 0:
                related_embeds = self.kg.get_embeddings_batch(related_cuis)
                # Combine direct and related embeddings
                all_embeds = torch.cat([cui_embeds, related_embeds], dim=0)
            else:
                all_embeds = cui_embeds
            
            # Take top-k most relevant (by norm as proxy for importance)
            if all_embeds.size(0) > top_k:
                norms = torch.norm(all_embeds, dim=-1)
                top_indices = torch.topk(norms, k=top_k).indices
                all_embeds = all_embeds[top_indices]
            
            knowledge_embeddings.append(all_embeds)
        
        return knowledge_embeddings
    
    def augment_with_knowledge(self, 
                               node_embeddings: torch.Tensor,
                               thoughts: List[ClinicalThought]) -> torch.Tensor:
        """
        Augment node embeddings with retrieved biomedical knowledge
        
        Args:
            node_embeddings: [num_nodes, hidden_dim]
            thoughts: List of ClinicalThought objects with linked entities
            
        Returns:
            Augmented embeddings [num_nodes, hidden_dim]
        """
        device = node_embeddings.device
        
        # Retrieve knowledge for each thought
        knowledge_embeds_list = self.retrieve_knowledge(thoughts, top_k=5)
        
        augmented_embeddings = []
        
        for idx, (node_emb, kg_embeds) in enumerate(zip(node_embeddings, knowledge_embeds_list)):
            kg_embeds = kg_embeds.to(device)
            
            if kg_embeds.size(0) == 0 or (kg_embeds.size(0) == 1 and torch.all(kg_embeds == 0)):
                # No knowledge available - use original embedding
                augmented_embeddings.append(node_emb)
                continue
            
            # Encode KG context
            kg_context = self.kg_context_encoder(kg_embeds)  # [num_kg, embed_dim]
            
            # Fuse via attention
            node_query = node_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
            kg_kv = kg_context.unsqueeze(0)  # [1, num_kg, embed_dim]
            
            attended, _ = self.knowledge_fusion(
                node_query,  # Query: original node
                kg_kv,       # Key: KG embeddings
                kg_kv        # Value: KG embeddings
            )
            
            attended = attended.squeeze(0).squeeze(0)  # [embed_dim]
            
            # Gated fusion
            gate_input = torch.cat([node_emb, attended], dim=-1)
            gate = self.fusion_gate(gate_input)
            
            # Combine: gate * attended + (1-gate) * original
            augmented = gate * attended + (1 - gate) * node_emb
            
            augmented_embeddings.append(augmented)
        
        return torch.stack(augmented_embeddings)


class MultiStageGoTReasoningEngine(nn.Module):
    """
    Module 2: Multi-Stage Graph-of-Thoughts Reasoning Engine
    """
    def __init__(self, base_llm, tokenizer, hidden_dim: int = 768):
        super().__init__()
        self.base_llm = base_llm
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        
        # Generation stage: diverse prompting strategies
        self.prompt_templates = [
            "Summarize the following clinical event focusing on the problem: {text}",
            "Describe the temporal progression of: {text}",
            "Explain the treatment and intervention in: {text}",
            "What is the clinical outcome described in: {text}"
        ]
        
        # Aggregation stage
        self.aggregation_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True
        )
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Refinement stage
        self.relevance_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.consistency_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    @property
    def device(self):
        return next(self.parameters()).device
        
    def generation_stage(self, thought: ClinicalThought, num_candidates: int = 4) -> List[str]:
        """Generate multiple candidate summaries"""
        candidates = []
        device = self.device
        
        for template in self.prompt_templates[:num_candidates]:
            prompt = template.format(text=thought.text)
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.base_llm.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=3,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            candidate = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            candidates.append(candidate)
        
        return candidates
    
    def aggregation_stage(self, node_embeddings: torch.Tensor, 
                         edge_index: torch.Tensor) -> torch.Tensor:
        """Aggregate information from connected nodes"""
        aggregated, _ = self.aggregation_attention(
            node_embeddings.unsqueeze(0),
            node_embeddings.unsqueeze(0),
            node_embeddings.unsqueeze(0)
        )
        
        combined = torch.cat([node_embeddings, aggregated.squeeze(0)], dim=-1)
        output = self.aggregation_mlp(combined)
        
        return output
    
    def refinement_stage(self, candidates: List[str], thought_embedding: torch.Tensor,
                        context_embeddings: Optional[torch.Tensor] = None) -> Tuple[str, float]:
        """Score and select best candidate"""
        best_score = -1.0
        best_candidate = candidates[0] if candidates else ""
        device = self.device
        
        for candidate in candidates:
            cand_inputs = self.tokenizer(candidate, return_tensors='pt', 
                                        padding=True, truncation=True)
            cand_inputs = {k: v.to(device) for k, v in cand_inputs.items()}
            
            with torch.no_grad():
                if hasattr(self.base_llm, 'get_encoder'):
                    cand_outputs = self.base_llm.get_encoder()(**cand_inputs)
                else:
                    cand_outputs = self.base_llm.encoder(**cand_inputs)
                cand_embedding = cand_outputs.last_hidden_state.mean(dim=1)
            
            relevance = self.relevance_scorer(cand_embedding).item()
            
            thought_emb = thought_embedding.unsqueeze(0) if thought_embedding.dim() == 1 else thought_embedding
            consistency_input = torch.cat([cand_embedding, thought_emb], dim=-1)
            consistency = self.consistency_scorer(consistency_input).item()
            
            if context_embeddings is not None and context_embeddings.numel() > 0:
                similarity = F.cosine_similarity(
                    cand_embedding, context_embeddings, dim=-1
                ).mean().item()
                redundancy_penalty = similarity
            else:
                redundancy_penalty = 0.0
            
            score = 0.4 * relevance + 0.4 * consistency - 0.2 * redundancy_penalty
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate, best_score
    
    def forward(self, thoughts: List[ClinicalThought], 
                node_embeddings: torch.Tensor,
                edge_index: torch.Tensor,
                num_iterations: int = 2) -> List[str]:
        """Execute complete GoT reasoning pipeline"""
        if len(thoughts) == 0:
            return []
            
        refined_summaries = []
        context_embeddings_list: List[torch.Tensor] = []
        device = self.device
        
        for iteration in range(num_iterations):
            iteration_summaries = []
            
            aggregated_embeddings = self.aggregation_stage(node_embeddings, edge_index)
            
            for idx, thought in enumerate(thoughts):
                candidates = self.generation_stage(thought)
                
                if context_embeddings_list:
                    context_tensor = torch.stack(context_embeddings_list)
                else:
                    context_tensor = None
                
                best_summary, score = self.refinement_stage(
                    candidates, 
                    aggregated_embeddings[idx],
                    context_tensor
                )
                
                iteration_summaries.append(best_summary)
                
                summary_inputs = self.tokenizer(best_summary, return_tensors='pt', 
                                               padding=True, truncation=True)
                summary_inputs = {k: v.to(device) for k, v in summary_inputs.items()}
                
                with torch.no_grad():
                    if hasattr(self.base_llm, 'get_encoder'):
                        summary_outputs = self.base_llm.get_encoder()(**summary_inputs)
                    else:
                        summary_outputs = self.base_llm.encoder(**summary_inputs)
                    summary_embedding = summary_outputs.last_hidden_state.mean(dim=1)
                context_embeddings_list.append(summary_embedding.squeeze(0))
            
            refined_summaries = iteration_summaries
        
        return refined_summaries


class HierarchicalDistillationLayer(nn.Module):
    """
    Module 4: Hierarchical Distillation for Multi-Granularity Summarization
    """
    def __init__(self, hidden_dim: int = 768, num_clusters: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        
        self.cluster_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_clusters),
            nn.Softmax(dim=-1)
        )
        
        self.token_importance = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.section_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=2
        )
        
        self.distillation_head = nn.Sequential(
            nn.Linear(hidden_dim * num_clusters, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    @property
    def device(self):
        return next(self.parameters()).device
        
    def cluster_nodes(self, node_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cluster nodes into clinical themes"""
        cluster_probs = self.cluster_predictor(node_embeddings)
        cluster_assignments = cluster_probs.argmax(dim=-1)
        return cluster_assignments, cluster_probs
    
    def generate_section_summaries(self, node_embeddings: torch.Tensor,
                                   cluster_assignments: torch.Tensor) -> torch.Tensor:
        """Generate summaries for each cluster"""
        section_embeddings = []
        device = self.device
        
        for cluster_id in range(self.num_clusters):
            mask = (cluster_assignments == cluster_id)
            if mask.sum() == 0:
                section_embeddings.append(torch.zeros(self.hidden_dim, device=device))
                continue
            
            cluster_nodes = node_embeddings[mask]
            importance_weights = self.token_importance(cluster_nodes)
            weighted_nodes = cluster_nodes * importance_weights
            
            section_repr = self.section_encoder(weighted_nodes.unsqueeze(0))
            section_summary = section_repr.mean(dim=1).squeeze(0)
            
            section_embeddings.append(section_summary)
        
        return torch.stack(section_embeddings)
    
    def distill_final_summary(self, section_embeddings: torch.Tensor) -> torch.Tensor:
        """Distill section summaries into final summary"""
        combined = section_embeddings.flatten()
        final_embedding = self.distillation_head(combined.unsqueeze(0))
        return final_embedding.squeeze(0)
    
    def forward(self, node_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Complete hierarchical distillation"""
        if node_embeddings.numel() == 0:
            device = self.device
            return (torch.zeros(self.hidden_dim, device=device), 
                    torch.zeros(self.num_clusters, self.hidden_dim, device=device))
        
        cluster_assignments, cluster_probs = self.cluster_nodes(node_embeddings)
        section_embeddings = self.generate_section_summaries(node_embeddings, cluster_assignments)
        final_embedding = self.distill_final_summary(section_embeddings)
        
        return final_embedding, section_embeddings


class GoTHCS(nn.Module):
    """
    Complete GoT-HCS Architecture with Real Knowledge Graph Integration
    """
    def __init__(self, 
                 base_model_name: str = "google/flan-t5-large",
                 umls_embeddings_path: Optional[str] = None,
                 umls_relations_path: Optional[str] = None):
        super().__init__()
        
        # Load base LLM
        from transformers import T5ForConditionalGeneration
        self.base_llm = T5ForConditionalGeneration.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Initialize Knowledge Graph
        print("Initializing Biomedical Knowledge Graph...")
        self.kg = BiomedicalKnowledgeGraph(
            umls_embeddings_path=umls_embeddings_path,
            umls_relations_path=umls_relations_path,
            embedding_dim=768
        )
        
        # Module 1: Graph Construction
        self.entity_extractor = TemporalClinicalEntityExtractor(kg=self.kg)
        self.graph_constructor = GraphConstructor()
        self.graph_encoder = ThoughtGraphEncoder()
        
        # Module 2: GoT Reasoning Engine
        self.reasoning_engine = MultiStageGoTReasoningEngine(
            self.base_llm, self.tokenizer
        )
        
        # Module 3: Knowledge Augmentation 
        self.knowledge_augmentation = KnowledgeGraphAugmentation(kg=self.kg)
        
        # Module 4: Hierarchical Distillation
        self.hierarchical_distillation = HierarchicalDistillationLayer()
        
        # Final generation head
        self.final_generator = nn.Linear(768, self.base_llm.config.vocab_size)
    
    @property
    def device(self):
        return next(self.parameters()).device
        
    def forward(self, clinical_notes: str, num_iterations: int = 2) -> Dict[str, Any]:
        """
        Complete forward pass through GoT-HCS
        """
        device = self.device
        
        # Step 1: Extract clinical thoughts and construct graph
        thoughts = self.entity_extractor.extract_entities(clinical_notes)
        
        if len(thoughts) == 0:
            return {
                'final_summary': "",
                'section_summaries': [],
                'thought_graph': {'nodes': [], 'edges': []},
                'num_thoughts': 0,
                'num_edges': 0,
                'kg_concepts': []
            }
        
        edges = self.graph_constructor.construct_edges(thoughts)
        
        # Prepare graph tensors
        node_features = torch.stack([t.embedding for t in thoughts]).to(device)
        
        if len(edges) > 0:
            edge_index = torch.tensor(
                [[e.source_idx for e in edges], [e.target_idx for e in edges]], 
                dtype=torch.long
            ).to(device)
            edge_weights = torch.tensor([e.weight for e in edges], dtype=torch.float).to(device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_weights = torch.empty(0, dtype=torch.float, device=device)
        
        # Step 2: Encode graph structure
        encoded_nodes = self.graph_encoder(node_features, edge_index, edge_weights)
        
        # Step 3: Augment with biomedical knowledge
        augmented_nodes = self.knowledge_augmentation.augment_with_knowledge(
            encoded_nodes, thoughts
        )
        
        # Step 4: Multi-stage GoT reasoning
        refined_summaries = self.reasoning_engine(
            thoughts, augmented_nodes, edge_index, num_iterations
        )
        
        # Step 5: Hierarchical distillation
        final_embedding, section_embeddings = self.hierarchical_distillation(augmented_nodes)
        
        # Step 6: Generate final summary
        summary_text = " ".join(refined_summaries[:3]) if refined_summaries else clinical_notes[:500]
        generation_prompt = f"Generate a brief hospital course summary: {summary_text}"
        inputs = self.tokenizer(generation_prompt, return_tensors='pt', truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.base_llm.generate(
                **inputs,
                max_new_tokens=300,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.8
            )
        
        final_summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Collect all linked KG concepts
        all_kg_concepts = []
        for thought in thoughts:
            all_kg_concepts.extend(thought.linked_entities)
        
        # Prepare output
        result = {
            'final_summary': final_summary,
            'section_summaries': refined_summaries,
            'thought_graph': {
                'nodes': [{'text': t.text, 'category': t.category, 'type': t.entity_type,
                          'kg_concepts': t.linked_entities} 
                         for t in thoughts],
                'edges': [{'source': e.source_idx, 'target': e.target_idx, 
                          'type': e.edge_type, 'weight': e.weight} 
                         for e in edges]
            },
            'num_thoughts': len(thoughts),
            'num_edges': len(edges),
            'kg_concepts': list(set(all_kg_concepts))  # Unique CUIs used
        }
        
        return result
    
    def generate_multi_granularity(self, clinical_notes: str) -> Dict[str, str]:
        """Generate summaries at multiple granularity levels"""
        device = self.device
        result = self.forward(clinical_notes)
        
        detailed = " ".join(result['section_summaries']) if result['section_summaries'] else ""
        standard = result['final_summary']
        
        brief_input = self.tokenizer(f"Summarize briefly: {standard}", return_tensors='pt')
        brief_input = {k: v.to(device) for k, v in brief_input.items()}
        
        with torch.no_grad():
            brief_output = self.base_llm.generate(
                **brief_input,
                max_new_tokens=100,
                num_beams=3
            )
        
        brief = self.tokenizer.decode(brief_output[0], skip_special_tokens=True)
        
        return {
            'detailed': detailed,
            'standard': standard,
            'brief': brief
        }


# Training utilities
class GoTHCSTrainer:
    """Training pipeline for GoT-HCS"""
    def __init__(self, model: GoTHCS, learning_rate: float = 2e-5):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
    def compute_loss(self, predictions: str, targets: str) -> torch.Tensor:
        """Multi-task loss"""
        device = self.model.device
        
        pred_tokens = self.model.tokenizer(predictions, return_tensors='pt', 
                                          padding=True, truncation=True)
        target_tokens = self.model.tokenizer(targets, return_tensors='pt',
                                            padding=True, truncation=True)
        pred_tokens = {k: v.to(device) for k, v in pred_tokens.items()}
        target_tokens = {k: v.to(device) for k, v in target_tokens.items()}
        
        pred_outputs = self.model.base_llm.get_encoder()(**pred_tokens)
        target_outputs = self.model.base_llm.get_encoder()(**target_tokens)
        
        pred_embeds = pred_outputs.last_hidden_state.mean(dim=1)
        target_embeds = target_outputs.last_hidden_state.mean(dim=1)
        
        summary_loss = 1 - F.cosine_similarity(pred_embeds, target_embeds.detach(), dim=-1).mean()
        
        return summary_loss
    
    def train_step(self, clinical_notes: str, target_summary: str) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        result = self.model(clinical_notes)
        loss = self.compute_loss(result['final_summary'], target_summary)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


# Visualization utilities
def visualize_thought_graph(thought_graph: Dict) -> str:
    """Generate DOT format for graph visualization"""
    dot_str = "digraph ThoughtGraph {\n"
    dot_str += "  rankdir=TB;\n"
    dot_str += "  node [shape=box, style=rounded];\n\n"
    
    for idx, node in enumerate(thought_graph['nodes']):
        label_text = node['text'][:50].replace('"', '\\"')
        kg_concepts = node.get('kg_concepts', [])
        kg_text = f"\\nKG: {', '.join(kg_concepts[:2])}" if kg_concepts else ""
        label = f"{node['category']}: {label_text}...{kg_text}"
        
        color = {
            'diagnostic': 'lightblue',
            'therapeutic': 'lightgreen',
            'monitoring': 'lightyellow',
            'outcome': 'lightpink'
        }.get(node['category'], 'white')
        
        dot_str += f'  {idx} [label="{label}", fillcolor={color}, style=filled];\n'
    
    dot_str += "\n"
    
    for edge in thought_graph['edges']:
        style = {
            'temporal': 'solid',
            'causal': 'bold',
            'logical': 'dashed'
        }.get(edge['type'], 'solid')
        
        dot_str += f'  {edge["source"]} -> {edge["target"]} [style={style}, label="{edge["type"]}"];\n'
    
    dot_str += "}\n"
    
    return dot_str


if __name__ == "__main__":
    print("="*80)
    print("GoT-HCS: Graph-of-Thoughts Enhanced Hierarchical Clinical Summarization")
    print("With Real Biomedical Knowledge Graph Integration")
    print("="*80)
    
    # Initialize model
    print("\nInitializing GoT-HCS model...")
    print("Note: For full functionality, provide paths to:")
    print("  - UMLS embeddings (e.g., from SapBERT, BioLORD, CODER)")
    print("  - UMLS relation triples (from UMLS Metathesaurus)")
    
    model = GoTHCS(
        base_model_name="google/flan-t5-base",
        umls_embeddings_path=None,  # Set to actual path when available
        umls_relations_path=None     # Set to actual path when available
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Example clinical note
    sample_note = """
    Patient admitted on 01/15/2024 with acute respiratory distress. 
    Initial vitals showed tachypnea and hypoxemia. Chest X-ray revealed bilateral infiltrates.
    Started on broad-spectrum antibiotics and supplemental oxygen on hospital day 1.
    Blood cultures obtained. Patient improved over next 48 hours.
    Antibiotics de-escalated based on culture results on hospital day 3.
    Discharged home on hospital day 5 with oral antibiotics.
    """
    
    print("\nProcessing clinical note...")
    with torch.no_grad():
        result = model(sample_note, num_iterations=2)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\n=== FINAL SUMMARY ===")
    print(result['final_summary'])
    
    print(f"\n=== GRAPH STATISTICS ===")
    print(f"Nodes: {result['num_thoughts']}")
    print(f"Edges: {result['num_edges']}")
    print(f"Linked KG Concepts: {len(result['kg_concepts'])}")
    if result['kg_concepts']:
        print(f"Sample CUIs: {result['kg_concepts'][:5]}")
    
    print("\n=== THOUGHT GRAPH VISUALIZATION ===")
    print(visualize_thought_graph(result['thought_graph']))
    
    print("\n=== MULTI-GRANULARITY SUMMARIES ===")
    with torch.no_grad():
        multi_gran = model.generate_multi_granularity(sample_note)
    print("Brief:", multi_gran['brief'])
    print("\nStandard:", multi_gran['standard'])
    print("\nDetailed:", multi_gran['detailed'][:200] + "..." if len(multi_gran['detailed']) > 200 else multi_gran['detailed'])
    
    # print("\n" + "="*80)
    # print("NOTES:")
    # print("- To use real UMLS embeddings, download from:")
    # print("  * SapBERT: https://github.com/cambridgeltl/sapbert")
    # print("  * BioLORD: https://github.com/FlagOpen/FlagEmbedding")
    # print("  * CODER: https://github.com/GanjinZero/CODER")
    # print("- UMLS can be obtained from: https://www.nlm.nih.gov/research/umls/")
    # print("="*80)


