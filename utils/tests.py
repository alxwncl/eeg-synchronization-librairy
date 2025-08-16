import sys
import pytest
import numpy as np
from io import StringIO

from channels_manager import reorder_synchronization_matrix


def test_basic_2d_reordering():
    """Test basic 2D matrix reordering functionality."""
    # Create a simple test matrix
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6], 
                       [7, 8, 9]])
    
    current_order = ['A', 'B', 'C']
    reference_order = ['C', 'A', 'B']
    
    expected = np.array([[9, 7, 8],
                         [3, 1, 2],
                         [6, 4, 5]])
    
    result = reorder_synchronization_matrix(matrix, current_order, reference_order)
    np.testing.assert_array_equal(result, expected)


def test_3d_reordering():
    """Test 3D array reordering (e.g., time series of connectivity matrices)."""
    # Create 3D test data: (time, electrodes, electrodes)
    matrix = np.random.rand(5, 3, 3)
    matrix[0] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    current_order = ['A', 'B', 'C']
    reference_order = ['C', 'A', 'B']
    
    result = reorder_synchronization_matrix(matrix, current_order, reference_order)
    
    # Check shape
    assert result.shape == (5, 3, 3)
    
    # Check first time slice reordering
    expected_first = np.array([[9, 7, 8], [3, 1, 2], [6, 4, 5]])
    np.testing.assert_array_equal(result[0], expected_first)


def test_4d_reordering():
    """Test 4D array reordering (e.g., subjects x conditions x electrodes x electrodes)."""
    matrix = np.random.rand(2, 3, 4, 4)
    current_order = ['A', 'B', 'C', 'D']
    reference_order = ['D', 'B', 'A', 'C']
    
    result = reorder_synchronization_matrix(matrix, current_order, reference_order)
    
    assert result.shape == (2, 3, 4, 4)
    
    # Verify that each slice is properly reordered
    for i in range(2):
        for j in range(3):
            original_slice = matrix[i, j]
            reordered_slice = result[i, j]
            
            # Check that diagonal elements moved correctly
            assert reordered_slice[0, 0] == original_slice[3, 3]  # D->D
            assert reordered_slice[1, 1] == original_slice[1, 1]  # B->B
            assert reordered_slice[2, 2] == original_slice[0, 0]  # A->A
            assert reordered_slice[3, 3] == original_slice[2, 2]  # C->C


def test_missing_electrodes():
    """Test handling of missing electrodes (should fill with NaN)."""
    matrix = np.array([[1, 2], [3, 4]])
    current_order = ['A', 'B']
    reference_order = ['A', 'C', 'B']  # 'C' is missing
    
    # Capture warning output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    result = reorder_synchronization_matrix(matrix, current_order, reference_order)
    
    # Restore stdout
    sys.stdout = old_stdout
    output = captured_output.getvalue()
    
    # Check warning was printed
    assert "Missing electrodes" in output
    assert "C" in output
    
    # Check result shape and NaN placement
    assert result.shape == (3, 3)
    assert result[0, 0] == 1  # A-A
    assert np.isnan(result[1, 1])  # C-C (missing)
    assert result[2, 2] == 4  # B-B
    assert np.isnan(result[0, 1])  # A-C (missing)
    assert result[0, 2] == 2  # A-B


def test_extra_electrodes():
    """Test handling of extra electrodes (should be ignored)."""
    matrix = np.array([[1, 2, 5], [3, 4, 6], [7, 8, 9]])
    current_order = ['A', 'B', 'X']  # 'X' is extra
    reference_order = ['B', 'A']
    
    # Capture warning output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    result = reorder_synchronization_matrix(matrix, current_order, reference_order)
    
    # Restore stdout  
    sys.stdout = old_stdout
    output = captured_output.getvalue()
    
    # Check warning was printed
    assert "Extra electrodes" in output
    assert "X" in output
    
    # Check result
    expected = np.array([[4, 3], [2, 1]])  # B-B, B-A, A-B, A-A
    np.testing.assert_array_equal(result, expected)


def test_identical_orders():
    """Test that identical orders return a copy of the original."""
    matrix = np.random.rand(10, 5, 5)
    electrode_order = ['A', 'B', 'C', 'D', 'E']
    
    result = reorder_synchronization_matrix(matrix, electrode_order, electrode_order)
    
    np.testing.assert_array_equal(result, matrix)
    # Ensure it's a copy, not the same object
    assert result is not matrix


def test_empty_reference_order():
    """Test behavior with empty reference order."""
    matrix = np.array([[1, 2], [3, 4]])
    current_order = ['A', 'B']
    reference_order = []
    
    result = reorder_synchronization_matrix(matrix, current_order, reference_order)
    
    assert result.shape == (0, 0)


def test_no_overlapping_electrodes():
    """Test case where no electrodes overlap."""
    matrix = np.array([[1, 2], [3, 4]])
    current_order = ['A', 'B']
    reference_order = ['C', 'D']
    
    result = reorder_synchronization_matrix(matrix, current_order, reference_order)
    
    assert result.shape == (2, 2)
    assert np.all(np.isnan(result))


def test_input_validation():
    """Test various input validation scenarios."""
    matrix = np.array([[1, 2], [3, 4]])
    current_order = ['A', 'B']
    reference_order = ['A', 'B']
    
    # Test 1D array (should fail)
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        reorder_synchronization_matrix(np.array([1, 2, 3]), current_order, reference_order)
    
    # Test mismatched electrode order length
    with pytest.raises(ValueError, match="doesn't match last matrix dimension"):
        reorder_synchronization_matrix(matrix, ['A'], reference_order)
    
    # Test non-square matrix
    non_square = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="must be equal"):
        reorder_synchronization_matrix(non_square, ['A', 'B', 'C'], ['A', 'B', 'C'])


def test_symmetric_matrix_property():
    """Test that symmetric matrices remain symmetric after reordering."""
    # Create symmetric matrix
    matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    current_order = ['A', 'B', 'C']
    reference_order = ['C', 'A', 'B']
    
    result = reorder_synchronization_matrix(matrix, current_order, reference_order)
    
    # Check symmetry is preserved
    np.testing.assert_array_equal(result, result.T)


def test_performance_with_large_arrays():
    """Test performance characteristics with larger arrays."""
    # Create larger test case
    n_electrodes = 64
    matrix = np.random.rand(100, n_electrodes, n_electrodes)  # 100 time points
    current_order = [f'E{i}' for i in range(n_electrodes)]
    reference_order = current_order[::-1]  # Reverse order
    
    result = reorder_synchronization_matrix(matrix, current_order, reference_order)
    
    assert result.shape == (100, n_electrodes, n_electrodes)
    
    # Verify reordering worked (check a few diagonal elements)
    assert result[0, 0, 0] == matrix[0, -1, -1]  # First becomes last
    assert result[0, -1, -1] == matrix[0, 0, 0]  # Last becomes first


def test_data_types_preservation():
    """Test that different data types are preserved."""
    # Test integer matrix
    int_matrix = np.array([[1, 2], [3, 4]], dtype=np.int32)
    current_order = ['A', 'B']
    reference_order = ['B', 'A']
    
    result = reorder_synchronization_matrix(int_matrix, current_order, reference_order)
    # Note: result will be float due to NaN filling capability, but actual values should be preserved
    assert result[0, 0] == 4
    assert result[1, 1] == 1


def run_all_tests():
    """Run all tests and report results."""
    test_functions = [
        test_basic_2d_reordering,
        test_3d_reordering,
        test_4d_reordering,
        test_missing_electrodes,
        test_extra_electrodes,
        test_identical_orders,
        test_empty_reference_order,
        test_no_overlapping_electrodes,
        test_input_validation,
        test_symmetric_matrix_property,
        test_performance_with_large_arrays,
        test_data_types_preservation
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: {str(e)}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    # Example usage
    print("Running comprehensive tests for matrix reordering functions...")
    all_passed = run_all_tests()
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed. Check the output above.")