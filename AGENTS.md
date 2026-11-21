# Repository instructions

## Security review before every push

Before pushing any branch or tag, review the exact commits and refs being sent. Compare them with the current remote state so the review covers all outgoing changes, including earlier local commits.

Check the security implications of the changed code and files. Pay particular attention to worker isolation, filesystem access, provider data sharing, credentials, dependencies, and release permissions when those areas change. For release tags, also verify the target commit and the workflow the tag triggers.

Report the review scope, findings, and checks run before pushing. Resolve blocking findings and review any subsequent changes before the push.
