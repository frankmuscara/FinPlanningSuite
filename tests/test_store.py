"""
Tests for client data storage and persistence.
"""
import pytest
import tempfile
import os
from pathlib import Path
from finplan_suite.core import store


@pytest.fixture
def temp_data_dir(monkeypatch):
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, "clients")
        monkeypatch.setattr(store, "DATA_DIR", test_dir)
        os.makedirs(test_dir, exist_ok=True)
        yield test_dir


@pytest.mark.unit
class TestClient:
    """Tests for Client dataclass."""

    def test_new_client_has_uuid(self):
        """New clients should get a valid UUID."""
        client = store.new_client()
        assert client.client_id is not None
        assert len(client.client_id) == 36  # UUID format
        assert "-" in client.client_id

    def test_new_client_defaults(self):
        """New clients should have sensible defaults."""
        client = store.new_client()
        assert client.first_name == ""
        assert client.last_name == ""
        assert client.retirement_age == 65
        assert client.filing_status == "Married Filing Jointly"
        assert client.state == "PA"
        assert client.accounts == []


@pytest.mark.unit
class TestClientPersistence:
    """Tests for saving and loading clients."""

    def test_save_and_load_client(self, temp_data_dir):
        """Should save and load client data correctly."""
        client = store.new_client()
        client.first_name = "John"
        client.last_name = "Doe"
        client.email = "john@example.com"
        client.birth_year = 1980
        client.retirement_age = 67

        store.save_client(client)

        loaded = store.load_client(client.client_id)
        assert loaded is not None
        assert loaded.client_id == client.client_id
        assert loaded.first_name == "John"
        assert loaded.last_name == "Doe"
        assert loaded.email == "john@example.com"
        assert loaded.birth_year == 1980
        assert loaded.retirement_age == 67

    def test_load_nonexistent_client(self, temp_data_dir):
        """Loading a client that doesn't exist should return None."""
        result = store.load_client("nonexistent-id")
        assert result is None

    def test_save_updates_timestamp(self, temp_data_dir):
        """Saving a client should update the updated_at timestamp."""
        client = store.new_client()
        client.first_name = "Jane"

        original_updated = client.updated_at
        store.save_client(client)

        loaded = store.load_client(client.client_id)
        # After save, updated_at should be set to today
        assert loaded.updated_at >= original_updated


@pytest.mark.unit
class TestClientListing:
    """Tests for listing and exporting clients."""

    def test_list_clients_empty(self, temp_data_dir):
        """Listing clients when none exist should return empty list."""
        clients = store.list_clients()
        assert clients == []

    def test_list_clients_multiple(self, temp_data_dir):
        """Should list all saved clients."""
        client1 = store.new_client()
        client1.first_name = "Alice"
        store.save_client(client1)

        client2 = store.new_client()
        client2.first_name = "Bob"
        store.save_client(client2)

        clients = store.list_clients()
        assert len(clients) == 2
        names = {c.first_name for c in clients}
        assert "Alice" in names
        assert "Bob" in names

    def test_list_clients_sorted_by_updated(self, temp_data_dir):
        """Clients should be sorted by updated_at descending."""
        import json
        import os

        # Create clients and manually modify their JSON files to have different dates
        client1 = store.new_client()
        client1.first_name = "First"
        store.save_client(client1)

        client2 = store.new_client()
        client2.first_name = "Second"
        store.save_client(client2)

        # Manually edit client1's JSON to have an older date
        path1 = os.path.join(temp_data_dir, f"{client1.client_id}.json")
        with open(path1, "r") as f:
            data1 = json.load(f)
        data1["updated_at"] = "2023-01-01"
        with open(path1, "w") as f:
            json.dump(data1, f, indent=2)

        # Manually edit client2's JSON to have a newer date
        path2 = os.path.join(temp_data_dir, f"{client2.client_id}.json")
        with open(path2, "r") as f:
            data2 = json.load(f)
        data2["updated_at"] = "2023-12-31"
        with open(path2, "w") as f:
            json.dump(data2, f, indent=2)

        clients = store.list_clients()
        # Most recently updated (2023-12-31) should be first
        assert clients[0].first_name == "Second"
        assert clients[1].first_name == "First"

    def test_export_clients_csv(self, temp_data_dir):
        """Should export clients to CSV format."""
        client = store.new_client()
        client.first_name = "Export"
        client.last_name = "Test"
        store.save_client(client)

        export_path = os.path.join(temp_data_dir, "export.csv")
        store.export_clients_csv(export_path)

        assert os.path.exists(export_path)

        # Verify CSV content
        with open(export_path, "r") as f:
            content = f.read()
            assert "Export" in content
            assert "Test" in content
            assert "client_id" in content  # Header
