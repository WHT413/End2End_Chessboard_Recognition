"""Main chess board detection and analysis pipeline."""

import cv2

from chess.utils.config import get_config
from core.chess_mapping import ChessMapper
from core.chess_processing import ChessProcessor
from core.display_manager import display_results, print_summary, show_board_comparison, process_chessboard_image
from core.results_builder import ResultsBuilder


def main():
    """Main pipeline for chess board detection and analysis."""
    # Initialize processors
    processor = ChessProcessor()
    mapper = ChessMapper()

    # Load configuration and get sample image
    config = get_config()
    img_path = processor.get_random_sample_image_path()
    proc_size = processor.get_processing_size()

    print(f"Processing image: {img_path}")

    try:
        # Initialize components
        board_detector, piece_detector, fen_converter, board_mapper, chess_visualizer = processor.initialize_components()

        # Load and detect board
        orig_img, corners, cropped_img = processor.load_and_detect_board(img_path)

        # Create separate copy for grid visualization
        grid_img = orig_img.copy()

        # Create homography mapping for coordinate transformation
        homography_data, grid_pts = mapper.create_homography_mapping(corners, orig_img, proc_size)

        # Detect pieces directly on original image using bottom-center reference
        piece_info, annotated_img = mapper.detect_pieces_on_original(piece_detector, orig_img)

        # Map piece coordinates from original image to board space
        mapped_pieces = mapper.map_pieces_to_board_space(piece_info, homography_data)

        # Assign pieces to 8x8 board matrix with conflict resolution
        mapping_result = mapper.assign_pieces_to_cells(mapped_pieces)

        # Create display coordinates for visualization
        orig_coords = mapper.create_display_coordinates(mapped_pieces)

        # Convert to FEN notation
        fen_string, fen_img = processor.convert_board_to_fen(mapping_result.chess_board)

        # Build analysis results
        results = (ResultsBuilder()
                   .with_detection_data(piece_info, orig_coords, annotated_img)
                   .with_board_data(mapping_result.chess_board, mapping_result.conf_matrix,
                                    mapping_result.conflict_positions)
                   .with_processing_data(mapping_result.debug_msgs, mapping_result.unmapped_pieces,
                                         mapping_result.conflict_msgs)
                   .with_viz_data(grid_img, cropped_img, grid_pts, fen_img)
                   .with_fen_string(fen_string)
                   .build())

        # Display results
        img_result, _ = display_results(results, chess_visualizer)

        # Show board comparison
        show_board_comparison(orig_img, fen_img)

        # Process and display comprehensive visualization
        process_chessboard_image(orig_img, annotated_img, img_result, fen_img)

        # Print summary
        print_summary(results)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
