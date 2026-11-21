class LegacyExportBuilder:
    def run(self) -> str:
        return "legacy"


class DeltaExportBuilder:
    def run(self) -> str:
        return "delta"


class DryRunExportBuilder:
    def run(self) -> str:
        return "dry-run"
