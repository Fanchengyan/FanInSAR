from pathlib import Path

import pytest

from faninsar._core.file_tools import (
    ensure_int,
    load_meta_value,
    load_meta_values,
    strip_str,
)


@pytest.fixture
def data_dir() -> Path:
    """Fixture for the data directory."""
    return Path(__file__).parent / "data" / "ascii"


@pytest.fixture
def ascii_file_center(data_dir: Path) -> Path:
    """Fixture for the center ASCII file."""
    return data_dir / "lower_left_center.asc"


@pytest.fixture
def ascii_file_corner(data_dir: Path) -> Path:
    """Fixture for the corner ASCII file."""
    return data_dir / "lower_left_corner.asc"

class TestLoadMetaValue:
    """Tests for the load_meta_value function."""

    def test_load_meta_value_success(
        self,
        ascii_file_center: Path,
        ascii_file_corner: Path,
    ) -> None:
        """Test successful retrieval of metadata values."""
        # test center file
        assert load_meta_value(ascii_file_center, "NCOLS") == "100"
        assert load_meta_value(ascii_file_center, "NROWS") == "200"
        assert load_meta_value(ascii_file_center, "XLLCENTER") == "300"
        assert load_meta_value(ascii_file_center, "YLLCENTER") == "400"
        # test corner file
        assert load_meta_value(ascii_file_corner, "NCOLS") == "100"
        assert load_meta_value(ascii_file_corner, "NROWS") == "200"
        assert load_meta_value(ascii_file_corner, "XLLCORNER") == "300"
        assert load_meta_value(ascii_file_corner, "YLLCORNER") == "400"

    def test_load_meta_value_case_insensitive(
        self,
        ascii_file_center: Path,
        ascii_file_corner: Path,
    ) -> None:
        """Test case-insensitive retrieval of metadata values."""
        # test center file
        assert load_meta_value(ascii_file_center, "ncols") == "100"
        assert load_meta_value(ascii_file_center, "nrows") == "200"
        assert load_meta_value(ascii_file_center, "xllcenter") == "300"
        assert load_meta_value(ascii_file_center, "yllcenter") == "400"
        # test corner file
        assert load_meta_value(ascii_file_corner, "ncols") == "100"
        assert load_meta_value(ascii_file_corner, "nrows") == "200"
        assert load_meta_value(ascii_file_corner, "xllcorner") == "300"
        assert load_meta_value(ascii_file_corner, "yllcorner") == "400"

    def test_load_meta_value_custom_separator(self, tmp_path: Path) -> None:
        """Test retrieval of metadata values with a custom separator."""
        file_path: Path = tmp_path / "test_meta_sep.asc"
        file_path.write_text("ncols:100\nnrows:200\n")
        with file_path.open() as f:
            print(30 * "=")
            print(f.readlines())
        assert load_meta_value(file_path, "ncols", sep=":") == "100"
        assert load_meta_value(file_path, "nrows", sep=":") == "200"

    def test_load_meta_value_line_range(self, tmp_path: Path) -> None:
        """Test retrieval of metadata values within a specific line range."""
        file_path: Path = tmp_path / "test_meta_range.asc"
        file_path.write_text("header\nncols 100\nnrows 200\n")
        assert load_meta_value(file_path, "ncols", line_start=1, line_end=3) == "100"
        assert load_meta_value(file_path, "nrows", line_start=1, line_end=3) == "200"

    def test_load_meta_value_invalid_line_range(self, ascii_file_center: Path) -> None:
        """Test retrieval of metadata values with an invalid line range."""
        with pytest.raises(ValueError):  # noqa: PT011
            load_meta_value(ascii_file_center, "ncols", line_start=3, line_end=1)

    def test_load_meta_value_invalid_line(self, ascii_file_center: Path) -> None:
        """Test retrieval of metadata values with an invalid line range."""
        with pytest.raises(ValueError):  # noqa: PT011
            load_meta_value(ascii_file_center, "ncols", line_start="a", line_end="b")

    def test_load_meta_value_key_not_found(self, ascii_file_center: Path) -> None:
        """Test retrieval of metadata values when the key is not found."""
        assert load_meta_value(ascii_file_center, "nonexistent") is None


class TestLoadMetaValues:
    """Tests for the load_meta_values function."""

    def test_load_meta_values_success(
        self,
        ascii_file_center: Path,
        ascii_file_corner: Path,
    ) -> None:
        """Test successful retrieval of multiple metadata values."""
        # test center file
        assert load_meta_values(ascii_file_center, ["NCOLS", "NROWS"]) == {
            "NCOLS": "100",
            "NROWS": "200",
        }
        assert load_meta_values(ascii_file_center, ["XLLCENTER", "YLLCENTER"]) == {
            "XLLCENTER": "300",
            "YLLCENTER": "400",
        }
        # test corner file
        assert load_meta_values(ascii_file_corner, ["NCOLS", "NROWS"]) == {
            "NCOLS": "100",
            "NROWS": "200",
        }
        assert load_meta_values(ascii_file_corner, ["XLLCORNER", "YLLCORNER"]) == {
            "XLLCORNER": "300",
            "YLLCORNER": "400",
        }

    def test_load_meta_values_case_insensitive(
        self,
        ascii_file_center: Path,
        ascii_file_corner: Path,
    ) -> None:
        """Test case-insensitive retrieval of multiple metadata values."""
        # test center file
        assert load_meta_values(ascii_file_center, ["ncols", "nrows"]) == {
            "ncols": "100",
            "nrows": "200",
        }
        assert load_meta_values(ascii_file_center, ["xllcenter", "yllcenter"]) == {
            "xllcenter": "300",
            "yllcenter": "400",
        }
        # test corner file
        assert load_meta_values(ascii_file_corner, ["ncols", "nrows"]) == {
            "ncols": "100",
            "nrows": "200",
        }
        assert load_meta_values(ascii_file_corner, ["xllcorner", "yllcorner"]) == {
            "xllcorner": "300",
            "yllcorner": "400",
        }

    def test_load_meta_values_custom_separator(self, tmp_path: Path) -> None:
        """Test retrieval of multiple metadata values with a custom separator."""
        file_path: Path = tmp_path / "test_meta_sep.asc"
        file_path.write_text("ncols: 100  \nnrows: 200  \n")
        assert load_meta_values(file_path, ["ncols", "nrows"], sep=":") == {
            "ncols": "100",
            "nrows": "200",
        }

    def test_load_meta_values_line_range(self, tmp_path: Path) -> None:
        """Test retrieval of multiple metadata values within a specific line range."""
        file_path: Path = tmp_path / "test_meta_range.asc"
        file_path.write_text("ncols 100\nnrows 200\nxllcenter 300\nyllcenter 400\n")
        assert load_meta_values(
            file_path,
            ["ncols", "nrows"],
            line_start=0,
            line_end=2,
        ) == {"ncols": "100", "nrows": "200"}
        assert load_meta_values(
            file_path,
            ["xllcenter", "yllcenter"],
            line_start=2,
            line_end=4,
        ) == {"xllcenter": "300", "yllcenter": "400"}

    def test_load_meta_values_invalid_line_range(self, ascii_file_center: Path) -> None:
        """Test handling of invalid line ranges."""
        with pytest.raises(ValueError):  # noqa: PT011
            load_meta_values(
                ascii_file_center,
                ["ncols", "nrows"],
                line_start=10,
                line_end=5,
            )

    def test_load_meta_values_key_not_found(self, ascii_file_center: Path) -> None:
        """Test behavior when keys are not found."""
        assert load_meta_values(ascii_file_center, ["invalid_key"]) == {
            "invalid_key": None,
        }


class TestEnsureInt:
    """Tests for the ensure_int function."""

    def test_ensure_int_valid(self) -> None:
        """Test valid integer conversion."""
        assert ensure_int("123", "test_value") == 123
        assert ensure_int("-456", "test_value") == -456
        assert ensure_int("0", "test_value") == 0

    def test_ensure_int_invalid(self) -> None:
        """Test invalid integer conversion raises ValueError."""
        with pytest.raises(ValueError, match="test_value must be an integer."):
            ensure_int("abc", "test_value")
        with pytest.raises(ValueError, match="test_value must be an integer."):
            ensure_int("123.45", "test_value")
        with pytest.raises(ValueError, match="test_value must be an integer."):
            ensure_int("", "test_value")

    def test_ensure_int_edge_cases(self) -> None:
        """Test edge cases for integer conversion."""
        assert ensure_int(str(2**31 - 1), "test_value") == 2**31 - 1  # Max 32-bit int
        assert ensure_int(str(-(2**31)), "test_value") == -(2**31)  # Min 32-bit int
        assert ensure_int(str(2**63 - 1), "test_value") == 2**63 - 1  # Max 64-bit int
        assert ensure_int(str(-(2**63)), "test_value") == -(2**63)  # Min 64-bit int


class TestStripStr:
    """Tests for the strip_str function."""

    def test_strip_whitespace(self) -> None:
        """Test stripping of leading and trailing whitespace."""
        assert strip_str("  hello  ") == "hello"
        assert strip_str("\thello\t") == "hello"
        assert strip_str("\nhello\n") == "hello"
        assert strip_str("\rhello\r") == "hello"

    def test_strip_quotes(self) -> None:
        """Test stripping of leading and trailing quotes."""
        assert strip_str('"hello"') == "hello"
        assert strip_str("'hello'") == "hello"
        assert strip_str(' "hello" ') == "hello"
        assert strip_str(" 'hello' ") == "hello"

    def test_strip_newline_carriage_return(self) -> None:
        """Test stripping of newline and carriage return characters."""
        assert strip_str("hello\n") == "hello"
        assert strip_str("hello\r") == "hello"
        assert strip_str("\nhello\r") == "hello"
        assert strip_str("\rhello\n") == "hello"

    def test_strip_combination(self) -> None:
        """Test stripping of a combination of characters."""
        assert strip_str(' \n"hello"\r ') == "hello"
        assert strip_str("\t'hello'\n") == "hello"
        assert strip_str("\r 'hello' \n") == "hello"
        assert strip_str(' \n "hello" \r ') == "hello"

    def test_no_stripping_needed(self) -> None:
        """Test strings that do not need any stripping."""
        assert strip_str("hello") == "hello"
        assert strip_str("world") == "world"
        assert strip_str("12345") == "12345"

    def test_empty_string(self) -> None:
        """Test an empty string."""
        assert strip_str("") == ""
