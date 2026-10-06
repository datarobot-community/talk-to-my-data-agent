### **GitHub Workflows**

#### 📌 **Overview**

This directory (`.github/workflows/`) contains GitHub Actions workflow definitions that automate various tasks such as
CI/CD, releases, and repository synchronization.

#### ⚙️ **Workflows**

Below is a list of the workflows included in this repository:

| Workflow File                  | Purpose                                                           |
|--------------------------------|-------------------------------------------------------------------|
| `license-check.yml`            | Check and fix license headers and resolve dependencies' licenses. |
| `python-static-checks.yml`     | Run Ruff linter and formatter, and MyPy static type checks.       |
| `python-deps-install-test.yml` | Verify Python dependencies install for different Python versions. |
| `shellcheck.yml`               | Run [shellcheck](https://github.com/koalaman/shellcheck/).        |
| `versions-lock.yml`            | Check `.datarobot/cli/versions.yaml` matches the `uv.lock` files. |
| `yaml-format.yml`              | Run YAML linter tool (yamlfmt).                                   |

---

Feel free to update this document as new workflows are added or modified! ✨
