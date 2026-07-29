from std.ffi import CStringSlice, external_call
from std.io.file import FileHandle
from std.os.path import dirname
from std.time import perf_counter_ns


def _c_string(value: String) raises -> String:
    """Returns owned nul-terminated storage accepted by CStringSlice."""
    if "\0" in value:
        raise Error("File paths cannot contain nul bytes.")
    return String(value, "\0")


def sync_file(mut file: FileHandle) raises:
    """Flushes file contents and metadata to stable storage."""
    if external_call["fsync", Int](file.handle) != 0:
        raise Error("fsync failed while saving the collection.")


def atomic_replace(source: String, destination: String) raises:
    """Atomically replaces destination with a file from the same directory."""
    var source_storage = _c_string(source)
    var destination_storage = _c_string(destination)
    var source_c = CStringSlice(source_storage)
    var destination_c = CStringSlice(destination_storage)
    if (
        external_call["rename", Int](
            source_c.unsafe_ptr(), destination_c.unsafe_ptr()
        )
        != 0
    ):
        raise Error("Atomic collection file replacement failed.")


def sync_parent_directory(path: String) raises:
    """Persists the directory entry installed by atomic_replace."""
    var parent = dirname(path)
    if parent.byte_length() == 0:
        parent = "."
    var parent_storage = _c_string(parent)
    var parent_c = CStringSlice(parent_storage)
    var directory = external_call["opendir", Int](parent_c.unsafe_ptr())
    if directory == 0:
        raise Error("Cannot open the collection parent directory.")
    var descriptor = external_call["dirfd", Int](directory)
    if descriptor < 0:
        _ = external_call["closedir", Int](directory)
        raise Error("Cannot access the collection parent directory.")
    var sync_result = external_call["fsync", Int](descriptor)
    var close_result = external_call["closedir", Int](directory)
    if sync_result != 0:
        raise Error("Cannot persist the collection directory entry.")
    if close_result != 0:
        raise Error("Cannot close the collection parent directory.")


def atomic_temporary_path(destination: String) -> String:
    """Creates a same-directory temporary name for one atomic publication."""
    var process_id = external_call["getpid", Int]()
    return String(
        destination,
        ".tmp.",
        process_id,
        ".",
        perf_counter_ns(),
    )
