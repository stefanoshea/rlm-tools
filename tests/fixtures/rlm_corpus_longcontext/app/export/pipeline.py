from app import feature_flags
from app.export.steps import DeltaExportBuilder, LegacyExportBuilder


def build_export_pipeline():
    if feature_flags.is_enabled(feature_flags.EXPORT_PIPELINE_V2):
        return DeltaExportBuilder()
    return LegacyExportBuilder()
