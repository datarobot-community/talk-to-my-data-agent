---
name: datarobot-app-framework-setup
description: Install and configure the official DataRobot agent skills, which help you design, develop, and deploy your DataRobot agent (including CI/CD setup via the datarobot-app-framework-cicd skill — GitHub Actions, GitLab CI, Pulumi IaC, secrets management). Use when setting up your agent to use the official DataRobot agent skills.
---

# Agent Skills

This project is best used with a variety of skills such as the
**datarobot-app-framework-cicd** skill for CI/CD setup (GitHub
Actions, GitLab CI, Pulumi IaC, secrets management), and the
datarobot-agent-assist skill for designing and simulating agent development.

Source: https://github.com/datarobot-oss/datarobot-agent-skills

## For agents: how to load the CI/CD skill and other highly relevant skills for this repository

1. **Check if already installed.** Look for
   `datarobot-app-framework-cicd` and `datarobot-agent-assist` in your
   available skills or installed plugins. If found, use it directly.

2. **If not installed**, install it using whichever method matches your runtime:

   - **Claude Code:**
     ```
     /plugin install datarobot-agent-skills
     ```

   - **Gemini CLI:**
     ```bash
     gemini skills install https://github.com/datarobot-oss/datarobot-agent-skills.git --consent
     ```

   - **Universal (any agent with npx):**
     ```bash
     npx ai-agent-skills install datarobot-oss/datarobot-agent-skills/
     ```

3. **Once loaded**, invoke the appropriate skill that was installed such as datarobot-app-framework-cicd to configure CI/CD pipelines or datarobot-agent-assist to design and simulate an agent for the user.

If installation fails, consult the upstream [installation guide](https://github.com/datarobot-oss/datarobot-agent-skills#quick-start).
