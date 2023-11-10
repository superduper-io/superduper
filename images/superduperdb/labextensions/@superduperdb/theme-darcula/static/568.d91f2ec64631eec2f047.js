"use strict";
(self.webpackChunk_superduperdb_theme_darcula = self.webpackChunk_superduperdb_theme_darcula || []).push([
   [568], {
      568: (e, a, t) => {
         Object.defineProperty(a, "__esModule", {
            value: !0
         });
         const l = {
            id: "@superduperdb/theme-darcula:plugin",
            requires: [t(643).IThemeManager],
            activate: function (e, a) {
               a.register({
                  name: "Darcula",
                  isLight: !1,
                  themeScrollbars: !0,
                  load: () => {
                    // Add the code to dynamically update the favicon
                    const updateFavicon = (faviconPath) => {
                      const linkElement = document.querySelector("link[rel='icon']");
                      if (linkElement) {
                        linkElement.href = faviconPath;
                      }
                    };

                    // Specify the path to your new favicon
                    const newFaviconPath = "./favicon.ico";

                    // Call the updateFavicon function with the new path
                    updateFavicon(newFaviconPath);

                     // Load CSS
                     a.loadCSS("@superduperdb/theme-darcula/index.css");
                  },
                  unload: () => {
                     // Unload theme
                     // You can also remove the favicon here if needed
                     Promise.resolve(void 0);
                  }
               })
            },
            autoStart: !0
         };
         a.default = l
      }
   }
]);