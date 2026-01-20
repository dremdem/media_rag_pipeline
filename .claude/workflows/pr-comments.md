# PR Discussion & Comment Management

When addressing PR feedback, there are two scenarios:

1. **In GitHub PR Comments**: When @claude is mentioned in a PR review comment/discussion on GitHub
2. **In Claude Code Session**: When the user asks you to address PR comments directly in a Claude Code session

Follow the appropriate guidelines below based on the context.

## Responding to PR Comments

1. **Read and Understand Context**
   - Read the entire comment thread to understand the discussion
   - Review related code changes and files
   - Identify the specific concern or question being raised

2. **Reply In-Thread**
   - To reply to a specific review comment thread, use the GitHub API:
     ```bash
     gh api repos/OWNER/REPO/pulls/PR_NUMBER/comments/COMMENT_ID/replies \
       -X POST -f body="Your threaded reply here"
     ```
   - To get the comment ID, fetch PR comments:
     ```bash
     gh api repos/OWNER/REPO/pulls/PR_NUMBER/comments
     ```
   - **Note**: `gh pr review --comment` creates top-level PR comments, NOT threaded replies
   - **NEVER mark comments as resolved** - let the reviewer resolve them
   - Keep replies focused and concise

3. **Comment Reply Format**
   ```markdown
   [Vary your greeting - examples: "Thanks for catching that!", "Good point!",
    "You're right!", "Agreed!", etc. - Acknowledge the specific point]

   [Explain what changes were made or will be made]

   Changes made in: `path/to/file.ext:line_number`
   Commit: [commit-hash]
   ```

4. **Making Changes**
   - Create **individual commits per comment** addressed
   - Use descriptive commit messages that reference the feedback
   - Push changes to the PR branch

5. **Commit Message Format for PR Comments**
   ```
   fix: [brief description addressing comment]

   Addresses PR comment: [link to comment or description]
   - [bullet point of change 1]
   - [bullet point of change 2]

   🤖 Generated with Claude Code
   Co-Authored-By: Claude <noreply@anthropic.com>
   ```

6. **Echo Summary**
   - After addressing comments, provide a summary in the Claude Code session or CI output
   - Include: number of comments addressed, files changed, and overall impact
   - Example: "Addressed 3 PR comments: updated error handling in `app/main.py:45`, fixed typo in docs, and added missing type hints"

## Best Practices

- Address comments promptly and thoroughly
- Ask clarifying questions if feedback is unclear
- Reference specific line numbers and file paths in responses
- Test changes locally before pushing
- Ensure CI checks pass after addressing comments
