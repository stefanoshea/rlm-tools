# Export Design Notes

Earlier drafts referenced `LegacyExportBuilder` because the migration to the
delta pipeline was not complete. The live code path now checks the export flag
in the application package before deciding which builder to instantiate.
