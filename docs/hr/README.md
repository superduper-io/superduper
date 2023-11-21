# SuperDuperDB Documentation Website

This documentation website is built using [Docusaurus 3.0](https://docusaurus.io/), a modern static website generator.

## Installation

As Docusaurus 3.0 lacks support for search functionality, we utilize Docusaurus 2.4 plugins. When installing, include the `--legacy-peer-deps` keyword to handle legacy peer dependencies.

```bash
npm install --legacy-peer-deps
```

## Local Development

To initiate a local development server and open a browser window, execute the following command. Most changes are reflected live without requiring a server restart.

```bash
npm run start
```

## Local Development with Search Feature

Since we employ local searching features, you need to build first and then serve it to see the result, as it builds its index during runtime.

```bash
npm run search
```

It will do both and show you a result. You have to run it again, if you want to see more changes.

## Build

Generate static content into the `build` directory using the following command. The output can be served using any static content hosting service.

```bash
npm run build
```

## Deployment

### Using SSH:

```bash
USE_SSH=true npm run deploy
```

### Without SSH:

```bash
GIT_USER=<Your GitHub username> npm run deploy
```

If you use GitHub Pages for hosting, this command conveniently builds the website and pushes it to the `gh-pages` branch.
