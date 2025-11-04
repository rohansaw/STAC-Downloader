import os
import tempfile
from unittest import mock

import pytest
import requests
from tenacity import RetryError

from stac_downloader.downloading import download_file


@pytest.fixture
def fake_url():
    return "http://example.com/testfile.txt"


@pytest.fixture
def fake_output_path(tmp_path):
    return tmp_path / "testfile.txt"


def test_download_file_success(fake_url, fake_output_path):
    mock_response = mock.Mock()
    mock_response.iter_content = lambda chunk_size: [b"test data"]
    mock_response.headers = {"Content-Length": str(len(b"test data"))}
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = mock.Mock()
    mock_response.raise_for_status = mock.Mock()

    with mock.patch("requests.get", return_value=mock_response):
        download_file(fake_url, str(fake_output_path))

    assert fake_output_path.exists()
    assert fake_output_path.read_bytes() == b"test data"


def test_download_file_skip_if_exists(fake_url, fake_output_path):
    fake_output_path.write_text("existing file")

    with mock.patch("requests.get") as mock_get:
        download_file(fake_url, str(fake_output_path), overwrite=False)
        mock_get.assert_not_called()
