# chat-with-your-docs

## Available Scripts

In the project directory, you can run the following npm scripts:

### Build for Production

Builds the application for production:

```bash
npm run build:production
```

This script will:

Build the application using the production configuration specified in webpack.config.js.
The production build is optimized for performance and should be used when deploying your application to a production environment.

### Build for development


Builds the application for development:

```bash
npm run build:development
```

This script will:

Build the application using the development configuration specified in webpack.config.js.
The development build includes additional debugging information and is suitable for local development and debugging.



### Serve Application
Serves the application using the serve command:

```bash
npm run serve
```

This script will:

Serve the contents of the dist/ directory, which is the output directory for the built application.
The application will be available at http://localhost:8080 (or the specified port if you change it).


### Start in Production Mode

Builds the application for production and serves it:   

```bash
npm run start:production
```
This script will:

Build the application for production.
Serve the production build on port 8080 using the serve command.
Use this script when you want to test your production-ready application.


### Start in Development Mode

Builds the application for development and serves it:

```bash
npm run start:development
```

This script will:

Build the application for development.
Serve the development build on port 8080 using the serve command.
Use this script when you want to test your application with debugging features enabled during local development.