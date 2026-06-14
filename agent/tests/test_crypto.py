"""Tests for crypto module."""

import tempfile
from pathlib import Path

import pytest

from huginn.crypto import CryptoVault, KeyManager, EncryptedDatabase, EncryptedConfig


class TestCryptoVault:
    def test_encrypt_decrypt_roundtrip(self):
        vault = CryptoVault("test_password")
        plaintext = "Hello, Huginn!"
        ciphertext = vault.encrypt(plaintext)
        assert ciphertext != plaintext.encode()
        decrypted = vault.decrypt(ciphertext)
        assert decrypted == plaintext

    def test_encrypt_bytes_roundtrip(self):
        vault = CryptoVault("test_password")
        data = b"\x00\x01\x02\xff" * 100
        ciphertext = vault.encrypt_bytes(data)
        decrypted = vault.decrypt_bytes(ciphertext)
        assert decrypted == data

    def test_different_salts_produce_different_ciphertexts(self):
        vault = CryptoVault("test_password")
        plaintext = "same text"
        c1 = vault.encrypt(plaintext)
        c2 = vault.encrypt(plaintext)
        assert c1 != c2  # Different salts
        assert vault.decrypt(c1) == vault.decrypt(c2)

    def test_lock_clears_key(self):
        vault = CryptoVault("test_password")
        assert vault.is_unlocked()
        vault.lock()
        assert not vault.is_unlocked()
        with pytest.raises(RuntimeError):
            vault.encrypt("test")

    def test_file_encryption(self):
        vault = CryptoVault("test_password")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "secret.txt"
            dst = Path(tmp) / "secret.enc"
            src.write_text("classified material data", encoding="utf-8")
            vault.encrypt_file(src, dst)
            assert dst.exists()
            assert dst.read_bytes() != src.read_bytes()

            out = Path(tmp) / "decrypted.txt"
            vault.decrypt_file(dst, out)
            assert out.read_text(encoding="utf-8") == "classified material data"

    def test_directory_encryption(self):
        vault = CryptoVault("test_password")
        with tempfile.TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "data"
            src_dir.mkdir()
            (src_dir / "file1.txt").write_text("data1")
            (src_dir / "subdir").mkdir()
            (src_dir / "subdir" / "file2.txt").write_text("data2")

            enc_path = Path(tmp) / "archive.enc"
            vault.encrypt_directory(src_dir, enc_path)

            out_dir = Path(tmp) / "restored"
            vault.decrypt_directory(enc_path, out_dir)
            assert (out_dir / "file1.txt").read_text() == "data1"
            assert (out_dir / "subdir" / "file2.txt").read_text() == "data2"


class TestKeyManager:
    def test_key_file_create_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "master.key"
            km = KeyManager(key_file)
            km.create_key_file("my_password", key_file)
            assert key_file.exists()

            km2 = KeyManager(key_file)
            key = km2.load_key_file("my_password")
            assert key is not None

            vault = km2.get_vault()
            assert vault.is_unlocked()
            ciphertext = vault.encrypt("test")
            assert vault.decrypt(ciphertext) == "test"

    def test_wrong_password_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "master.key"
            km = KeyManager(key_file)
            km.create_key_file("correct", key_file)
            km2 = KeyManager(key_file)
            with pytest.raises(Exception):
                km2.load_key_file("wrong")


class TestEncryptedDatabase:
    def test_mount_unmount(self):
        vault = CryptoVault("db_password")
        with tempfile.TemporaryDirectory() as tmp:
            enc_path = Path(tmp) / "db.sqlite3.enc"
            db = EncryptedDatabase(vault, enc_path)

            with db as plaintext_path:
                # Simulate database creation
                plaintext_path.parent.mkdir(parents=True, exist_ok=True)
                plaintext_path.write_text("sqlite3 data")

            assert enc_path.exists()
            assert not db.is_mounted()

            with db as plaintext_path:
                assert plaintext_path.read_text() == "sqlite3 data"

    def test_unmount_without_save_on_exception(self):
        vault = CryptoVault("db_password")
        with tempfile.TemporaryDirectory() as tmp:
            enc_path = Path(tmp) / "db.sqlite3.enc"
            db = EncryptedDatabase(vault, enc_path)

            try:
                with db as plaintext_path:
                    plaintext_path.parent.mkdir(parents=True, exist_ok=True)
                    plaintext_path.write_text("should_not_save")
                    raise ValueError("abort")
            except ValueError:
                pass

            assert not enc_path.exists()


class TestEncryptedConfig:
    def test_save_load_roundtrip(self):
        vault = CryptoVault("cfg_password")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = EncryptedConfig(config_path=Path(tmp) / "config.enc", vault=vault)
            config = {"api_key": "secret123", "provider": "anthropic"}
            cfg.save(config)
            assert cfg.exists()

            loaded = cfg.load()
            assert loaded == config
