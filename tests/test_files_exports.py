import pytest


@pytest.mark.unit
def test_files_api_exports():
    from fastapi_openai_compat import FileObject, create_files_router
    from fastapi_openai_compat.files import FileObject as ModuleFileObject
    from fastapi_openai_compat.files import create_files_router as module_create_files_router

    assert callable(create_files_router)
    assert callable(module_create_files_router)
    assert FileObject is ModuleFileObject
