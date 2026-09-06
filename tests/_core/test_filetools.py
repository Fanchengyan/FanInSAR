from pathlib import Path

import pytest

from faninsar.io.file_tools import ensure_int, load_meta, load_metas, strip_str


@pytest.fixture
def data_dir() -> Path:
    """Fixture for the data directory."""
    return Path(__file__).parents[1] / "data" / "ascii"


@pytest.fixture
def ascii_file_center(data_dir: Path) -> Path:
    """Fixture for the center ASCII file."""
    return data_dir / "lower_left_center.asc"


@pytest.fixture
def ascii_file_corner(data_dir: Path) -> Path:
    """Fixture for the corner ASCII file."""
    return data_dir / "lower_left_corner.asc"


@pytest.fixture
def create_test_file(tmp_path: Path):
    """Fixture for creating test files."""
    def _create_file(filename: str, content: str) -> Path:
        file_path = tmp_path / filename
        file_path.write_text(content)
        return file_path
    return _create_file


class TestLoadMeta:
    """Tests for the load_meta function."""

    @pytest.mark.parametrize("file_fixture,key,expected", [
        ("ascii_file_center", "NCOLS", "100"),
        ("ascii_file_center", "NROWS", "200"),
        ("ascii_file_center", "XLLCENTER", "300"),
        ("ascii_file_center", "YLLCENTER", "400"),
        ("ascii_file_corner", "NCOLS", "100"),
        ("ascii_file_corner", "NROWS", "200"),
        ("ascii_file_corner", "XLLCORNER", "300"),
        ("ascii_file_corner", "YLLCORNER", "400"),
    ])
    def test_load_meta_success(self, request, file_fixture: str, key: str, expected: str) -> None:
        """Test successful retrieval of metadata values."""
        file_path = request.getfixturevalue(file_fixture)
        assert load_meta(file_path, key) == expected

    @pytest.mark.parametrize("file_fixture,key,expected", [
        ("ascii_file_center", "ncols", "100"),
        ("ascii_file_center", "nrows", "200"),
        ("ascii_file_center", "xllcenter", "300"),
        ("ascii_file_center", "yllcenter", "400"),
        ("ascii_file_corner", "ncols", "100"),
        ("ascii_file_corner", "nrows", "200"),
        ("ascii_file_corner", "xllcorner", "300"),
        ("ascii_file_corner", "yllcorner", "400"),
    ])
    def test_load_meta_case_insensitive(self, request, file_fixture: str, key: str, expected: str) -> None:
        """Test case-insensitive retrieval of metadata values."""
        file_path = request.getfixturevalue(file_fixture)
        assert load_meta(file_path, key) == expected

    def test_load_meta_custom_separator(self, create_test_file) -> None:
        """Test retrieval of metadata values with a custom separator."""
        file_path = create_test_file("test_meta_sep.asc", "ncols:100\nnrows:200\n")
        assert load_meta(file_path, "ncols", sep=":") == "100"
        assert load_meta(file_path, "nrows", sep=":") == "200"

    def test_load_meta_line_range(self, create_test_file) -> None:
        """Test retrieval of metadata values within a specific line range."""
        file_path = create_test_file("test_meta_range.asc", "header\nncols 100\nnrows 200\n")
        assert load_meta(file_path, "ncols", line_start=1, line_end=3) == "100"
        assert load_meta(file_path, "nrows", line_start=1, line_end=3) == "200"

    def test_load_meta_invalid_line_range(self, ascii_file_center: Path) -> None:
        """Test behavior with invalid line range."""
        with pytest.raises(ValueError, match="line_start must be less than or equal to line_end"):
            load_meta(ascii_file_center, "ncols", line_start=3, line_end=1)

    def test_load_meta_invalid_line_type(self, ascii_file_center: Path) -> None:
        """Test behavior with invalid line type values."""
        with pytest.raises(TypeError):
            # Type annotation indicates that passing non-integer types raises TypeError
            load_meta(ascii_file_center, "ncols", line_start="a", line_end=1)  # type: ignore

        with pytest.raises(TypeError):
            # Type annotation indicates that passing non-integer types raises TypeError
            load_meta(ascii_file_center, "ncols", line_start=1, line_end="b")  # type: ignore

    def test_load_meta_key_not_found(self, ascii_file_center: Path) -> None:
        """Test behavior when key is not found."""
        assert load_meta(ascii_file_center, "nonexistent") is None


class TestLoadMetas:
    """Tests for the load_metas function."""

    @pytest.mark.parametrize("file_fixture,keys,expected", [
        (
            "ascii_file_center",
            ["NCOLS", "NROWS"],
            {"NCOLS": "100", "NROWS": "200"}
        ),
        (
            "ascii_file_center",
            ["XLLCENTER", "YLLCENTER"],
            {"XLLCENTER": "300", "YLLCENTER": "400"}
        ),
        (
            "ascii_file_corner",
            ["NCOLS", "NROWS"],
            {"NCOLS": "100", "NROWS": "200"}
        ),
        (
            "ascii_file_corner",
            ["XLLCORNER", "YLLCORNER"],
            {"XLLCORNER": "300", "YLLCORNER": "400"}
        ),
    ])
    def test_load_metas_success(self, request, file_fixture: str, keys: list, expected: dict) -> None:
        """Test successful retrieval of multiple metadata values."""
        file_path = request.getfixturevalue(file_fixture)
        assert load_metas(file_path, keys) == expected

    @pytest.mark.parametrize("file_fixture,keys,expected", [
        (
            "ascii_file_center",
            ["ncols", "nrows"],
            {"ncols": "100", "nrows": "200"}
        ),
        (
            "ascii_file_center",
            ["xllcenter", "yllcenter"],
            {"xllcenter": "300", "yllcenter": "400"}
        ),
        (
            "ascii_file_corner",
            ["ncols", "nrows"],
            {"ncols": "100", "nrows": "200"}
        ),
        (
            "ascii_file_corner",
            ["xllcorner", "yllcorner"],
            {"xllcorner": "300", "yllcorner": "400"}
        ),
    ])
    def test_load_metas_case_insensitive(self, request, file_fixture: str, keys: list, expected: dict) -> None:
        """Test case-insensitive retrieval of multiple metadata values."""
        file_path = request.getfixturevalue(file_fixture)
        assert load_metas(file_path, keys) == expected

    def test_load_metas_custom_separator(self, create_test_file) -> None:
        """Test retrieval of multiple metadata values with a custom separator."""
        file_path = create_test_file("test_meta_sep.asc", "ncols: 100  \nnrows: 200  \n")
        assert load_metas(file_path, ["ncols", "nrows"], sep=":") == {
            "ncols": "100",
            "nrows": "200",
        }

    def test_load_metas_line_range(self, create_test_file) -> None:
        """Test retrieval of multiple metadata values within a specific line range."""
        content = "ncols 100\nnrows 200\nxllcenter 300\nyllcenter 400\n"
        file_path = create_test_file("test_meta_range.asc", content)

        assert load_metas(
            file_path,
            ["ncols", "nrows"],
            line_start=0,
            line_end=2,
        ) == {"ncols": "100", "nrows": "200"}

        assert load_metas(
            file_path,
            ["xllcenter", "yllcenter"],
            line_start=2,
            line_end=4,
        ) == {"xllcenter": "300", "yllcenter": "400"}

    def test_load_metas_invalid_line_range(self, ascii_file_center: Path) -> None:
        """Test handling of invalid line ranges."""
        with pytest.raises(ValueError, match="line_start must be less than or equal to line_end"):
            load_metas(
                ascii_file_center,
                ["ncols", "nrows"],
                line_start=10,
                line_end=5,
            )

    def test_load_metas_key_not_found(self, ascii_file_center: Path) -> None:
        """Test behavior when keys are not found."""
        assert load_metas(ascii_file_center, ["invalid_key"]) == {
            "invalid_key": None,
        }


class TestEnsureInt:
    """Tests for the ensure_int function."""

    @pytest.mark.parametrize("value,name,expected", [
        ("123", "test_value", 123),
        ("-456", "test_value", -456),
        ("0", "test_value", 0),
        (str(2**31 - 1), "test_value", 2**31 - 1),  # Max 32-bit int
        (str(-(2**31)), "test_value", -(2**31)),  # Min 32-bit int
        (str(2**63 - 1), "test_value", 2**63 - 1),  # Max 64-bit int
        (str(-(2**63)), "test_value", -(2**63)),  # Min 64-bit int
    ])
    def test_ensure_int_valid(self, value: str, name: str, expected: int) -> None:
        """Test valid integer conversion."""
        assert ensure_int(value, name) == expected

    @pytest.mark.parametrize("value,name,error_msg", [
        ("abc", "test_value", "test_value must be an integer."),
        ("123.45", "test_value", "test_value must be an integer."),
        ("", "test_value", "test_value must be an integer."),
    ])
    def test_ensure_int_invalid(self, value: str, name: str, error_msg: str) -> None:
        """Test invalid integer conversion raises ValueError."""
        with pytest.raises(TypeError, match=error_msg):
            ensure_int(value, name)


class TestStripStr:
    """Tests for the strip_str function."""

    @pytest.mark.parametrize("input_str,expected", [
        ("  hello  ", "hello"),
        ("\thello\t", "hello"),
        ("\nhello\n", "hello"),
        ("\rhello\r", "hello"),
        ('"hello"', "hello"),
        ("'hello'", "hello"),
        (' "hello" ', "hello"),
        (" 'hello' ", "hello"),
        ("hello\n", "hello"),
        ("hello\r", "hello"),
        ("\nhello\r", "hello"),
        ("\rhello\n", "hello"),
        (' \n"hello"\r ', "hello"),
        ("\t'hello'\n", "hello"),
        ("\r 'hello' \n", "hello"),
        (' \n "hello" \r ', "hello"),
        ("hello", "hello"),
        ("world", "world"),
        ("", ""),
    ])
    def test_strip_str(self, input_str: str, expected: str) -> None:
        """Test string stripping functionality."""
        assert strip_str(input_str) == expected
