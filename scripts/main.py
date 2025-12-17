# Main Script - JPMC Census Income Classification & Segmentation Project

import argparse
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from classification_model import main as run_classification
from segmentation_model import main as run_segmentation


def main():
    parser = argparse.ArgumentParser(
        description='JPMC Census Income Classification & Segmentation'
    )
    parser.add_argument(
        '--classification-only',
        action='store_true',
        help='Run only classification model training'
    )
    parser.add_argument(
        '--segmentation-only',
        action='store_true',
        help='Run only customer segmentation'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("JPMC CENSUS INCOME PROJECT")
    print("Classification & Customer Segmentation")
    print("=" * 80)
    
    results = {}
    
    # Run classification
    if not args.segmentation_only:
        print("\n>>> STEP 1: CLASSIFICATION MODEL <<<")
        try:
            models, classification_results = run_classification()
            results['classification'] = {
                'models': models,
                'metrics': classification_results
            }
            print("\nClassification completed successfully")
        except Exception as e:
            print(f"\nClassification failed: {e}")
            import traceback
            traceback.print_exc()
            if args.classification_only:
                return
    
    # Run segmentation
    if not args.classification_only:
        print("\n>>> STEP 2: CUSTOMER SEGMENTATION <<<")
        try:
            profile_df, segments = run_segmentation()
            results['segmentation'] = {
                'profiles': profile_df,
                'labels': segments
            }
            print("\nSegmentation completed successfully")
        except Exception as e:
            print(f"\nSegmentation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    if 'classification' in results:
        print("\nClassification Models Trained:")
        for model_name in results['classification']['models'].keys():
            if results['classification']['models'][model_name] is not None:
                print(f"  {model_name}")
    
    if 'segmentation' in results:
        print(f"\nSegmentation:")
        print(f"  {len(results['segmentation']['profiles'])} segments created")
    
    print("\nOutputs saved to:")
    print(f"  Models: {Path(__file__).parent.parent / 'models'}")
    print(f"  Results: {Path(__file__).parent.parent / 'outputs'}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
