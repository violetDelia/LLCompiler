class Importer:
    """
    import model to MLIR module.
    """

    def __init__(self, *args, **kwargs) -> None:
        return

    def importer(self, model, *args, **kwargs):
        """
        import model to MLIR module.
        """
        raise NotImplementedError

    def _importer_torch_module(self, model, *args, **kwargs):
        raise NotImplementedError
    
    def _importer_fx_module(self, model, *args, **kwargs):
        raise NotImplementedError

    def _importer_onnx(self, model, *args, **kwargs):
        raise NotImplementedError