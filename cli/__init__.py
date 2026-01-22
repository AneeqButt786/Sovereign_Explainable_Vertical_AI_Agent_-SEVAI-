"""
CLI Prototype - Command Line Interface for SEVAI
Simple interface for testing the reasoning pipeline
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from agents.orchestrator import get_orchestrator
from core.logging_config import setup_logging, get_logger
from core.config import get_settings

# Initialize logging
setup_logging()
logger = get_logger("cli")


def format_output(result: dict, format_type: str = "text") -> str:
    """
    Format reasoning result for display
    
    Args:
        result: Reasoning result from orchestrator
        format_type: Output format (text or json)
        
    Returns:
        Formatted string
    """
    if format_type == "json":
        return json.dumps(result, indent=2)
    
    # Text format
    output_lines = []
    output_lines.append("=" * 60)
    output_lines.append("SEVAI - Medical Reasoning Analysis")
    output_lines.append("=" * 60)
    output_lines.append("")
    
    # Summary
    output_lines.append(f"Conclusion: {result['conclusion']}")
    output_lines.append(f"Confidence: {result['confidence']:.2%}")
    output_lines.append(f"Duration: {result['total_duration_ms']:.2f}ms")
    output_lines.append("")
    
    # Risk Flags
    if result['risk_flags']:
        output_lines.append("⚠️  Risk Flags:")
        for flag in result['risk_flags']:
            output_lines.append(f"  - {flag}")
        output_lines.append("")
    
    # Agent Results
    output_lines.append("Agent Execution Summary:")
    output_lines.append("-" * 60)
    
    for agent_name, agent_data in result['agent_results'].items():
        output_lines.append(f"\n{agent_name.replace('_', ' ').title()}:")
        output_lines.append(f"  Confidence: {agent_data['confidence']:.2%}")
        output_lines.append(f"  Duration: {agent_data['duration_ms']:.2f}ms")
        
        # Show key outputs
        output = agent_data['output']
        if isinstance(output, dict):
            if 'causal_chains' in output:
                chains = output['causal_chains']
                output_lines.append(f"  Causal Chains: {len(chains)}")
                for i, chain in enumerate(chains[:3], 1):  # Show first 3
                    output_lines.append(
                        f"    {i}. {chain.get('from', '')} → {chain.get('to', '')} "
                        f"(confidence: {chain.get('confidence', 0):.2f})"
                    )
            elif 'contradictions' in output:
                contradictions = output['contradictions']
                output_lines.append(f"  Contradictions Found: {len(contradictions)}")
                for i, contradiction in enumerate(contradictions, 1):
                    output_lines.append(f"    {i}. {contradiction.get('type', 'unknown')}: severity {contradiction.get('severity', 'unknown')}")
    
    output_lines.append("")
    output_lines.append("=" * 60)
    
    return "\n".join(output_lines)


def process_file(file_path: Path, output_format: str = "text") -> None:
    """
    Process input from file
    
    Args:
        file_path: Path to input file
        output_format: Output format (text or json)
    """
    logger.info(f"Processing file: {file_path}")
    
    try:
        # Read input
        with open(file_path, 'r', encoding='utf-8') as f:
            input_text = f.read()
        
        # Get orchestrator
        orchestrator = get_orchestrator()
        
        # Execute pipeline
        result = orchestrator.execute_pipeline(
            input_text=input_text,
            source=str(file_path),
            metadata={"input_file": str(file_path)}
        )
        
        # Format and display output
        formatted = format_output(result, output_format)
        print(formatted)
        
        logger.info("Processing complete")
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


def process_text(text: str, output_format: str = "text") -> None:
    """
    Process direct text input
    
    Args:
        text: Input text
        output_format: Output format (text or json)
    """
    logger.info("Processing text input")
    
    try:
        # Get orchestrator
        orchestrator = get_orchestrator()
        
        # Execute pipeline
        result = orchestrator.execute_pipeline(
            input_text=text,
            source="cli",
            metadata={"input_type": "direct_text"}
        )
        
        # Format and display output
        formatted = format_output(result, output_format)
        print(formatted)
        
        logger.info("Processing complete")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


def export_trail(execution_id: int, output_file: Optional[Path] = None) -> None:
    """
    Export reasoning trail to JSON
    
    Args:
        execution_id: Execution ID to export
        output_file: Output file path (optional)
    """
    logger.info(f"Exporting reasoning trail for execution {execution_id}")
    
    try:
        orchestrator = get_orchestrator()
        trail = orchestrator.get_reasoning_trail(execution_id)
        
        trail_json = json.dumps(trail, indent=2)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(trail_json)
            logger.info(f"Trail exported to: {output_file}")
        else:
            print(trail_json)
        
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="SEVAI - Sovereign Explainable Vertical AI Agent (Medical Domain)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process text file
  python -m cli.main --file case.txt
  
  # Process direct text
  python -m cli.main --text "Patient presents with fever and cough..."
  
  # Export reasoning trail
  python -m cli.main --export-trail 123 --output trail.json
  
  # Output as JSON
  python -m cli.main --file case.txt --format json
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file', '-f',
        type=Path,
        help='Input file path'
    )
    input_group.add_argument(
        '--text', '-t',
        type=str,
        help='Direct text input'
    )
    input_group.add_argument(
        '--export-trail', '-e',
        type=int,
        metavar='EXECUTION_ID',
        help='Export reasoning trail for execution ID'
    )
    
    # Output options
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file path (for trail export)'
    )
    
    args = parser.parse_args()
    
    # Route to appropriate handler
    if args.file:
        process_file(args.file, args.format)
    elif args.text:
        process_text(args.text, args.format)
    elif args.export_trail is not None:
        export_trail(args.export_trail, args.output)


if __name__ == "__main__":
    main()
