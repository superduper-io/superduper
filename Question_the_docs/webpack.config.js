const path = require('path');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const webpack = require('webpack');
const TerserPlugin = require('terser-webpack-plugin');

module.exports = {
  entry: path.join(__dirname, './src/index.ts'), // Entry point for your TypeScript widget component
  output: {
    path: path.resolve(__dirname, 'dist'), // Output directory for the bundled file
    filename: 'widget.js', // Name of the bundled file
    library: 'WidgetComponent', // Name of the global variable exposed
    libraryTarget: 'umd', // Universal Module Definition
  },
  resolve: {
    extensions: ['.ts', '.tsx', '.js'],
  },
  module: {
    rules: [
      {
        test: /\.(ts|tsx)$/, // Match TypeScript files
        exclude: /node_modules/,
        use: {
          loader: 'ts-loader', // Transpile TypeScript using ts-loader
        },
      },
      {
        test: /\.(js|jsx)$/, // Match JavaScript and JSX files
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader', // Transpile JavaScript and JSX using babel-loader
          options: {
            presets: ['@babel/preset-env', '@babel/preset-react'],
          },
        },
      },
      {
        test: /\.css$/, // Add this rule to handle CSS files
        exclude: /\.module\.css$/, // Exclude CSS Modules files
        use: ['style-loader', 'css-loader'],
      },
      {
        test: /\.module\.css$/, // Handle CSS Modules files
        use: ['style-loader', { loader: 'css-loader', options: { modules: true } }],
      },
    ],
  },
  optimization: {
    minimize: true, // Enable minification
    minimizer: [
      new TerserPlugin({
        // Terser options go here within terserOptions
        terserOptions: {
          sourceMap: true, // Enable source maps for debugging
          compress: {
            drop_console: true, // Remove console.log statements
          },
        },
      }),
    ],
  },
  plugins: [
    new CopyWebpackPlugin({
      patterns: [
        {
          from: 'public', // Source directory
          to: '.',      // Destination directory
        },
      ],
    }),
    new webpack.BannerPlugin({
      banner: `var divElement = document.createElement('div');
      divElement.id = 'superduperdb_app';
      document.body.appendChild(divElement);
      const data = document.getElementById('my-api');
      const collectionName = data.getAttribute('data-api-key');
      var currentScript = document.currentScript;
      if (collectionName !== 'superduperdb'&& collectionName !== 'langchain'&& collectionName !== 'huggingface') {
        if (currentScript) {
          currentScript.parentNode.removeChild(currentScript);
          console.log('Current script tag removed because the variable name is not "superduperdb".');
      } else {
          console.log('Current script tag not found.');
      }
      }
      `,
      raw: true,
    }),
    
    
  ],
};
