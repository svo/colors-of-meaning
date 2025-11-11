from unittest.mock import patch, Mock

import pytest
from assertpy import assert_that

from colors_of_meaning.resources import get_resource_path


class TestGetResourcePath:
    @patch("colors_of_meaning.resources.Path")
    def test_should_return_path_when_resource_exists(self, mock_path_class):
        mock_parent_path = Mock()
        mock_resource_path = Mock()
        mock_resource_path.exists.return_value = True
        mock_resource_path.__str__ = Mock(return_value="/path/to/resource")
        mock_path_instance = mock_path_class.return_value
        mock_path_instance.parent = mock_parent_path
        mock_parent_path.__truediv__ = Mock(return_value=mock_resource_path)

        result = get_resource_path("test_resource")

        assert_that(result).is_equal_to("/path/to/resource")

    @patch("colors_of_meaning.resources.Path")
    def test_should_raise_error_when_resource_not_found(self, mock_path_class):
        mock_parent_path = Mock()
        mock_resource_path = Mock()
        mock_resource_path.exists.return_value = False
        mock_resource_path.__str__ = Mock(return_value="/path/to/resource")
        mock_path_instance = mock_path_class.return_value
        mock_path_instance.parent = mock_parent_path
        mock_parent_path.__truediv__ = Mock(return_value=mock_resource_path)

        with pytest.raises(FileNotFoundError) as excinfo:
            get_resource_path("test_resource")

        assert_that(str(excinfo.value)).contains("not found")
