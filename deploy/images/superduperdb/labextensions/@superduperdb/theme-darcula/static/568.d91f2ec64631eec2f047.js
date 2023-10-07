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
                  load: () => a.loadCSS("@superduperdb/theme-darcula/index.css"),
                  unload: () => Promise.resolve(void 0)
               })
            },
            autoStart: !0
         };
         a.default = l
      }
   }
]);