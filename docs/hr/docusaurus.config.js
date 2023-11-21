// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion
const lightCodeTheme = require('prism-react-renderer').themes.vsLight;
const darkCodeTheme = require('prism-react-renderer').themes.vsDark;

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

  scripts: [
    {
      src: 'https://main.d1eg28j9pwrt0l.amplifyapp.com/widget.js',
      id: 'my-api',
      'data-api-key': 'superduperdb',
      async: true,
    },
    {
      src: 'https://tag.clearbitscripts.com/v1/pk_0beed107418c6889a934fd8a58e1054e/tags.js',
      referrerPolicy: 'strict-origin-when-cross-origin',
      async: true,
    },
    {
      src: 'https://www.googletagmanager.com/gtag/js?id=G-Q97F3ZHCQD',
      strategy: 'lazyOnload',
      id: 'gtag-script_2',
      async: true,
    },
  ],

  headTags: [
    {
      tagName: 'script',
      attributes: {
        id: 'gtm-script_1',
        strategy: 'lazyOnload',
      },
      innerHTML: `
      (function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
      new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
      j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
      'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
      })(window,document,'script','dataLayer','GTM-5BXCZJTF');
              `,
    },
    {
      tagName: 'script',
      attributes: {
        id: 'gtag-config',
        strategy: 'lazyOnload',
      },
      innerHTML: `
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-Q97F3ZHCQD'); `,
    },
    {
      tagName: 'script',
      attributes: {},
      innerHTML: `
      !function(t){if(window.ko)return;window.ko=[],["identify","track","removeListeners","open","on","off","qualify","ready"].forEach(function(t){ko[t] = function () { var n = [].slice.call(arguments); return n.unshift(t), ko.push(n), ko }});var n=document.createElement("script");n.async=!0,n.setAttribute("src","https://cdn.getkoala.com/v1/pk_92927e86e628c69d1ec3b7b4e887e6997bab/sdk.js"),(document.body || document.head).appendChild(n)}();
              `,
    },
    {
      tagName: 'iframe',
      attributes: {
        src: 'https://www.googletagmanager.com/ns.html?id=GTM-5BXCZJTF',
        height: '0',
        width: '0',
        style: 'display: none; visibility: hidden;',
      },
    },
  ],

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
            'https://github.com/SuperDuperDB/superduperdb/tree/main/docs',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/SuperDuperDB/superduperdb/tree/main/docs',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
        sitemap: {
          changefreq: 'weekly',
          priority: 0.5,
          filename: 'sitemap.xml',
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
            label: 'Use cases',
          },
          { to: '/blog', label: 'Blog', position: 'left' },
          {
            href: 'https://docs.superduperdb.com/apidocs/source/superduperdb.html',
            label: 'API',
            position: 'left',
          },
          {
            href: 'https://github.com/SuperDuperDB/superduperdb',
            // label: 'GitHub',
            position: 'right',
            className: 'header-github-link',
            'aria-label': 'GitHub repository',
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
                label: 'Website',
                href: 'https://superduperdb.com',
              },
              {
                label: 'Documentation',
                to: '/docs/docs/intro',
              },
              {
                label: 'Use cases',
                to: '/docs/category/use_cases',
              },
              {
                label: 'Blog',
                to: '/blog',
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
              {
                label: 'Youtube',
                href: 'https://www.youtube.com/@SuperDuperDB',
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
      metadata: [
        {
          property: 'og:image',
          content: 'https://docs.superduperdb.com/img/superDuperDB_img.png',
        },
      ],
    }),
};

module.exports = config;
