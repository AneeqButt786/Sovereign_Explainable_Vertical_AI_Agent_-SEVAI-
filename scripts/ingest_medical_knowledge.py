"""
Medical Knowledge Ingestion Script
Ingests medical guidelines and documents into Pinecone Vector Store
"""

import sys
import os
import argparse
from typing import List, Dict, Any
import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.vector_store import get_vector_store
from core.logging_config import get_logger

logger = get_logger("ingestion_script")


def ingest_file(filepath: str, source_name: str, topic: str):
    """Ingest a single file"""
    logger.info(f"Ingesting file: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        vector_store = get_vector_store()
        
        # Prepare document
        document = {
            "text": text,
            "metadata": {
                "source": source_name,
                "topic": topic,
                "filename": os.path.basename(filepath),
                "type": "medical_guideline"
            }
        }
        
        # Upsert
        stats = vector_store.upsert_documents([document])
        logger.info(f"Ingestion complete: {stats}")
        
    except Exception as e:
        logger.error(f"Failed to ingest file {filepath}: {e}")


def ingest_sample_knowledge():
    """Ingest sample medical knowledge for testing"""
    logger.info("Ingesting sample medical knowledge...")
    
    samples = [
        {
            "text": """
            Community-acquired pneumonia (CAP) is a common condition. 
            Symptoms include fever, cough with sputum, chest pain, and difficulty breathing.
            Diagnosis is confirmed by chest X-ray showing consolidation (infiltrates).
            Common pathogens include Streptococcus pneumoniae, Haemophilus influenzae.
            First-line treatment for healthy adults is amoxicillin or doxycycline.
            For patients with comorbidities, a breakdown of respiratory fluoroquinolone or combination therapy is recommended.
            Expected outcome with treatment is improvement within 48-72 hours.
            """,
            "source": "Clinical Guidelines 2024",
            "topic": "Pneumonia"
        },
        {
            "text": """
            Type 2 Diabetes Mellitus management guidelines.
            Diagnosis based on HbA1c >= 6.5% or Fasting Plasma Glucose >= 126 mg/dL.
            Symptoms include polyuria, polydipsia, unexplained weight loss.
            First-line pharmacotherapy is Metformin.
            Lifestyle modifications (diet, exercise) are critical.
            Complications include neuropathy, retinopathy, nephropathy.
            """,
            "source": "Endocrine Society Guidelines",
            "topic": "Diabetes"
        },
        {
            "text": """
            Hypertension (High Blood Pressure) guidelines.
            Stage 1 Hypertension: 130-139 / 80-89 mmHg.
            Stage 2 Hypertension: >= 140 / 90 mmHg.
            Diagnosis requires multiple readings on separate occasions.
            Initial treatment depends on stage and cardiovascular risk.
            First-line agents: Thiazide diuretics, ACE inhibitors, ARBs, Calcium channel blockers.
            Goal: < 130/80 mmHg.
            """,
            "source": "Cardiology Guidelines 2024",
            "topic": "Hypertension"
        }
    ]
    
    vector_store = get_vector_store()
    
    documents = []
    for sample in samples:
        documents.append({
            "text": sample["text"],
            "metadata": {
                "source": sample["source"],
                "topic": sample["topic"],
                "type": "guideline_summary"
            }
        })
    
    stats = vector_store.upsert_documents(documents)
    logger.info(f"Sample ingestion stats: {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest medical knowledge into Vector Store")
    parser.add_argument("--file", help="Path to text file to ingest")
    parser.add_argument("--source", help="Source name (e.g., 'WHO Guidelines')", default="Manual Import")
    parser.add_argument("--topic", help="Medical topic", default="General")
    parser.add_argument("--sample", action="store_true", help="Ingest sample medical knowledge")
    
    args = parser.parse_args()
    
    if args.sample:
        ingest_sample_knowledge()
    elif args.file:
        ingest_file(args.file, args.source, args.topic)
    else:
        parser.print_help()
