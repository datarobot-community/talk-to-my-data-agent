package trivy

import data.lib.trivy

default ignore := false

# litellm CVE-2026-49468 (CRITICAL) is in the LiteLLM proxy authentication
# layer. This app uses litellm only as an SDK client through
# instructor.from_litellm(litellm.acompletion) and does not run the LiteLLM
# proxy server, so the vulnerable code path is not reachable. Bumping to the
# fixed LiteLLM line requires OpenAI v2, while this repo is still pinned to
# openai<2.
ignore {
    input.PkgName == "litellm"
    input.VulnerabilityID == "CVE-2026-49468"
}

# starlette CVE-2026-48818, CVE-2026-54283 (HIGH) — SSRF and DoS via form
# parsing. The fix requires starlette>=1.x, a major version bump from 0.52
# that is not yet compatible with the fastapi version this project uses.
# Track for upgrade when fastapi ships a 1.x-compatible release.
ignore {
    input.PkgName == "starlette"
    input.VulnerabilityID == "CVE-2026-48818"
}

ignore {
    input.PkgName == "starlette"
    input.VulnerabilityID == "CVE-2026-54283"
}

# litellm CVE-2026-35030, CVE-2026-35029, CVE-2026-42271,
# CVE-2026-47101, CVE-2026-47102 are in LiteLLM proxy/gateway HTTP
# endpoints and proxy auth/authorization flows. This app uses litellm only
# as an SDK client through instructor.from_litellm(litellm.acompletion) and
# does not run the LiteLLM proxy server. The fixed LiteLLM line also requires
# openai>=2, while this repo is pinned to openai<2.
ignore {
    input.PkgName == "litellm"
    input.VulnerabilityID == "CVE-2026-35030"
}

ignore {
    input.PkgName == "litellm"
    input.VulnerabilityID == "CVE-2026-35029"
}

ignore {
    input.PkgName == "litellm"
    input.VulnerabilityID == "CVE-2026-42271"
}

ignore {
    input.PkgName == "litellm"
    input.VulnerabilityID == "CVE-2026-47101"
}

ignore {
    input.PkgName == "litellm"
    input.VulnerabilityID == "CVE-2026-47102"
}

ignore {
    input.PkgName == "litellm"
    input.VulnerabilityID == "GHSA-69x8-hrgq-fjj8"
}

# pyarrow CVE-2026-25087 (HIGH) cannot be upgraded narrowly: current
# datarobot-genai 0.3.x/0.4.x pins pyarrow==21.0.0, while datarobot-genai
# 0.5.x is a broader dependency migration. Track for upgrade when this app
# can move to the newer datarobot-genai line.
ignore {
    input.PkgName == "pyarrow"
    input.VulnerabilityID == "CVE-2026-25087"
}
