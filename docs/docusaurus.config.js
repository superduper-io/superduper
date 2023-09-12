// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'SuperDuperDB documentation',
  tagline: 'Bringing AI to your data-store',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://docs.superduperdb.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'SuperDuperDB', // Usually your GitHub org/user name.
  projectName: 'superduperdb', // Usually your repo name.

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: 'docs',
          path: 'content',
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/SuperDuperDB/superduperdb',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/SuperDuperDB/superduperdb',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      navbar: {
        title: 'SuperDuperDB',
        logo: {
          alt: 'My Site Logo',
          src: 'img/logo.png',
          href: 'https://superduperdb.com',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Documentation',
          },
          {
            type: 'docSidebar',
            sidebarId: 'useCasesSidebar',
            position: 'left',
            label: 'Use-Cases',
            
          },
          {to: '/blog', label: 'Blog', position: 'left'},
          {
            href: 'https://docs.superduperdb.com/apidocs/source/superduperdb.html',
            label: 'API',
            position: 'left',
          },
          {
            href: 'https://github.com/SuperDuperDB/superduperdb',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Resources',
            items: [
              {
                label: 'Documentation',
                to: '/docs/docs/intro',
              },
              {
                label: 'Use-Cases',
                to: '/docs/category/use_cases',
              },
              {
                label: 'Blog',
                to: '/blog',
              },
              {
                label: 'Youtube',
                to: 'https://www.youtube.com/@SuperDuperDB',
              },
            ],
          },
          {
            title: 'Project',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/SuperDuperDB/superduperdb',
              },
              {
                label: 'Issues',
                href: 'https://github.com/SuperDuperDB/superduperdb/issues',
              },
              {
                label: 'Discussions',
                href: 'https://github.com/SuperDuperDB/superduperdb/discussions',
              },
              {
                label: 'Roadmap',
                href: 'https://github.com/orgs/SuperDuperDB/projects/1/views/10',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'LinkedIn',
                href: 'https://www.linkedin.com/company/superduperdb/',
              },
              {
                label: 'Slack',
                href: 'https://superduperdb.slack.com/',
              },
              {
                label: 'X / Twitter',
                href: 'https://twitter.com/superduperdb',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} SuperDuperDB, Inc.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
