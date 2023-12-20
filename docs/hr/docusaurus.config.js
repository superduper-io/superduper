// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion
const lightCodeTheme = require('prism-react-renderer').themes.github;
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

  plugins: [
    [
      require.resolve('@cmfcmf/docusaurus-search-local'),
      {
        // whether to index docs pages
        indexDocs: true,

        // Whether to also index the titles of the parent categories in the sidebar of a doc page.
        // 0 disables this feature.
        // 1 indexes the direct parent category in the sidebar of a doc page
        // 2 indexes up to two nested parent categories of a doc page
        // 3...
        //
        // Do _not_ use Infinity, the value must be a JSON-serializable integer.
        indexDocSidebarParentCategories: 0,

        // whether to index blog pages
        indexBlog: true,

        // whether to index static pages
        // /404.html is never indexed
        indexPages: false,

        // language of your documentation, see next section
        language: 'en',

        // setting this to "none" will prevent the default CSS to be included. The default CSS
        // comes from autocomplete-theme-classic, which you can read more about here:
        // https://www.algolia.com/doc/ui-libraries/autocomplete/api-reference/autocomplete-theme-classic/
        // When you want to overwrite CSS variables defined by the default theme, make sure to suffix your
        // overwrites with `!important`, because they might otherwise not be applied as expected. See the
        // following comment for more information: https://github.com/cmfcmf/docusaurus-search-local/issues/107#issuecomment-1119831938.
        style: undefined,
        // https://github.com/algolia/autocomplete/blob/next/packages/autocomplete-theme-classic/src/theme.scss

        // The maximum number of search results shown to the user. This does _not_ affect performance of
        // searches, but simply does not display additional search results that have been found.
        maxSearchResults: 8,

        // lunr.js-specific settings
        lunr: {
          // When indexing your documents, their content is split into "tokens".
          // Text entered into the search box is also tokenized.
          // This setting configures the separator used to determine where to split the text into tokens.
          // By default, it splits the text at whitespace and dashes.
          //
          // Note: Does not work for "ja" and "th" languages, since these use a different tokenizer.
          tokenizerSeparator: /[\s\-]+/,
          // https://lunrjs.com/guides/customising.html#similarity-tuning
          //
          // This parameter controls the importance given to the length of a document and its fields. This
          // value must be between 0 and 1, and by default it has a value of 0.75. Reducing this value
          // reduces the effect of different length documents on a term’s importance to that document.
          b: 0.75,
          // This controls how quickly the boost given by a common word reaches saturation. Increasing it
          // will slow down the rate of saturation and lower values result in quicker saturation. The
          // default value is 1.2. If the collection of documents being indexed have high occurrences
          // of words that are not covered by a stop word filter, these words can quickly dominate any
          // similarity calculation. In these cases, this value can be reduced to get more balanced results.
          k1: 1.2,
          // By default, we rank pages where the search term appears in the title higher than pages where
          // the search term appears in just the text. This is done by "boosting" title matches with a
          // higher value than content matches. The concrete boosting behavior can be controlled by changing
          // the following settings.
          titleBoost: 5,
          contentBoost: 1,
          tagsBoost: 3,
          parentCategoriesBoost: 2, // Only used when indexDocSidebarParentCategories > 0
        },
      },
    ],
  ],
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
          // sidebarCollapsible: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/SuperDuperDB/superduperdb/blob/main/docs/hr',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/SuperDuperDB/superduperdb/blob/main/docs/hr',
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
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: '',
        logo: {
          alt: 'SuperDuperDB',
          src: 'img/SuperDuperDB_logo_color_dark_noborder.svg',
          srcDark: 'img/SuperDuperDB_logo_color.svg',
          href: 'https://superduperdb.com',
          target: '_self',
          // width: 250,
          // height: 34,
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Documentation',
          },
          { to: '/blog', label: 'Blog', position: 'left' },
          {
            href: 'https://docs.superduperdb.com/apidocs/source/superduperdb.html',
            label: 'API',
            position: 'left',
          },
          {
            href: 'https://github.com/SuperDuperDB/superduper-community-apps/',
            label: 'Community Apps',
            position: 'left',
          },
          {
            href: 'https://www.question-the-docs.superduperdb.com/',
            label: 'Ask our docs',
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
                to: '/docs/category/get-started',
              },
              {
                label: 'Use cases',
                to: '/docs/category/use-cases',
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
              // {
              //   label: 'Roadmap',
              //   href: 'https://github.com/orgs/SuperDuperDB/projects/1/views/10',
              // },
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
      announcementBar: {
        id: 'support_us',
        content: '🔮 We officially released v0.1 on Github! 🔮',
        backgroundColor: '#7628f8',
        textColor: '#fff',
        isCloseable: true,
      },
    }),
};

module.exports = config;
