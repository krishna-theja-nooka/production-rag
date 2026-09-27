# Publish this repository on GitHub

Suggested repository name: **production-rag**.
Suggested description: **Local-first RAG backend with hybrid retrieval, cited answers,
evaluation, tests, and in-process Python inference.**

The project is ready to commit. It has not been published to a GitHub account unless you
have separately connected GitHub and confirmed a successful push.

## Browser upload

1. Sign in to https://github.com and choose **New repository**.
2. Name it `production-rag`; choose public for a portfolio or private while learning.
3. If uploading this complete project, leave GitHub's README/license/gitignore options empty.
4. Upload the extracted project contents, not the ZIP itself. Include `.github`,
   `.gitignore` and `.env.example`; some file pickers hide dotfiles.
5. Confirm `.env`, local databases, and private documents are not included.
6. Open **Actions** to check CI after the push. Enable Actions if your account requires it.

## Git commands (recommended)

Run from the extracted project folder. Replace `YOUR_USERNAME` with your GitHub username.

```bash
git init -b main
git add .
git status --short
git commit -m "Build end-to-end local-first RAG reference application"
git remote add origin https://github.com/YOUR_USERNAME/production-rag.git
git push -u origin main
```

Authenticate with GitHub's supported browser/credential flow. Never paste a personal
access token into this README, code, or a chat message. If Git asks for author identity,
configure your own Git name and email locally before committing.

After publishing, add topics such as `rag`, `fastapi`, `retrieval-augmented-generation`,
`transformers`, `sentence-transformers`, and `python`. Link the repo from your portfolio.
Hosting source on GitHub does not run the chatbot server.
