var _JUPYTERLAB;
(() => {
   "use strict";
   var e, r, t, n, a, o, i, u, l, s, f, d, c, p, h, v = {
         254: (e, r, t) => {
            var n = {
                  "./index": () => t.e(568).then((() => () => t(568))),
                  "./extension": () => t.e(568).then((() => () => t(568)))
               },
               a = (e, r) => (t.R = r, r = t.o(n, e) ? n[e]() : Promise.resolve().then((() => {
                  throw new Error('Module "' + e + '" does not exist in container.')
               })), t.R = void 0, r),
               o = (e, r) => {
                  if (t.S) {
                     var n = "default",
                        a = t.S[n];
                     if (a && a !== e) throw new Error("Container initialization failed as it has already been initialized with a different share scope");
                     return t.S[n] = e, t.I(n, r)
                  }
               };
            t.d(r, {
               get: () => a,
               init: () => o
            })
         }
      },
      m = {};

   function g(e) {
      var r = m[e];
      if (void 0 !== r) return r.exports;
      var t = m[e] = {
         exports: {}
      };
      return v[e](t, t.exports, g), t.exports
   }
   g.m = v, g.c = m, g.d = (e, r) => {
      for (var t in r) g.o(r, t) && !g.o(e, t) && Object.defineProperty(e, t, {
         enumerable: !0,
         get: r[t]
      })
   }, g.f = {}, g.e = e => Promise.all(Object.keys(g.f).reduce(((r, t) => (g.f[t](e, r), r)), [])), g.u = e => e + ".d91f2ec64631eec2f047.js?v=d91f2ec64631eec2f047", g.g = function () {
      if ("object" == typeof globalThis) return globalThis;
      try {
         return this || new Function("return this")()
      } catch (e) {
         if ("object" == typeof window) return window
      }
   }(), g.o = (e, r) => Object.prototype.hasOwnProperty.call(e, r), e = {}, r = "@superduperdb/theme-darcula:", g.l = (t, n, a, o) => {
      if (e[t]) e[t].push(n);
      else {
         var i, u;
         if (void 0 !== a)
            for (var l = document.getElementsByTagName("script"), s = 0; s < l.length; s++) {
               var f = l[s];
               if (f.getAttribute("src") == t || f.getAttribute("data-webpack") == r + a) {
                  i = f;
                  break
               }
            }
         i || (u = !0, (i = document.createElement("script")).charset = "utf-8", i.timeout = 120, g.nc && i.setAttribute("nonce", g.nc), i.setAttribute("data-webpack", r + a), i.src = t), e[t] = [n];
         var d = (r, n) => {
               i.onerror = i.onload = null, clearTimeout(c);
               var a = e[t];
               if (delete e[t], i.parentNode && i.parentNode.removeChild(i), a && a.forEach((e => e(n))), r) return r(n)
            },
            c = setTimeout(d.bind(null, void 0, {
               type: "timeout",
               target: i
            }), 12e4);
         i.onerror = d.bind(null, i.onerror), i.onload = d.bind(null, i.onload), u && document.head.appendChild(i)
      }
   }, (() => {
      g.S = {};
      var e = {},
         r = {};
      g.I = (t, n) => {
         n || (n = []);
         var a = r[t];
         if (a || (a = r[t] = {}), !(n.indexOf(a) >= 0)) {
            if (n.push(a), e[t]) return e[t];
            g.o(g.S, t) || (g.S[t] = {});
            var o = g.S[t],
               i = "@superduperdb/theme-darcula",
               u = [];
            return "default" === t && ((e, r, t, n) => {
               var a = o[e] = o[e] || {},
                  u = a[r];
               (!u || !u.loaded && (1 != !u.eager ? n : i > u.from)) && (a[r] = {
                  get: () => g.e(568).then((() => () => g(568))),
                  from: i,
                  eager: !1
               })
            })("@superduperdb/theme-darcula", "4.0.0"), e[t] = u.length ? Promise.all(u).then((() => e[t] = 1)) : 1
         }
      }
   })(), (() => {
      var e;
      g.g.importScripts && (e = g.g.location + "");
      var r = g.g.document;
      if (!e && r && (r.currentScript && (e = r.currentScript.src), !e)) {
         var t = r.getElementsByTagName("script");
         t.length && (e = t[t.length - 1].src)
      }
      if (!e) throw new Error("Automatic publicPath is not supported in this browser");
      e = e.replace(/#.*$/, "").replace(/\?.*$/, "").replace(/\/[^\/]+$/, "/"), g.p = e
   })(), t = e => {
      var r = e => e.split(".").map((e => +e == e ? +e : e)),
         t = /^([^-+]+)?(?:-([^+]+))?(?:\+(.+))?$/.exec(e),
         n = t[1] ? r(t[1]) : [];
      return t[2] && (n.length++, n.push.apply(n, r(t[2]))), t[3] && (n.push([]), n.push.apply(n, r(t[3]))), n
   }, n = (e, r) => {
      e = t(e), r = t(r);
      for (var n = 0;;) {
         if (n >= e.length) return n < r.length && "u" != (typeof r[n])[0];
         var a = e[n],
            o = (typeof a)[0];
         if (n >= r.length) return "u" == o;
         var i = r[n],
            u = (typeof i)[0];
         if (o != u) return "o" == o && "n" == u || "s" == u || "u" == o;
         if ("o" != o && "u" != o && a != i) return a < i;
         n++
      }
   }, a = e => {
      var r = e[0],
         t = "";
      if (1 === e.length) return "*";
      if (r + .5) {
         t += 0 == r ? ">=" : -1 == r ? "<" : 1 == r ? "^" : 2 == r ? "~" : r > 0 ? "=" : "!=";
         for (var n = 1, o = 1; o < e.length; o++) n--, t += "u" == (typeof (u = e[o]))[0] ? "-" : (n > 0 ? "." : "") + (n = 2, u);
         return t
      }
      var i = [];
      for (o = 1; o < e.length; o++) {
         var u = e[o];
         i.push(0 === u ? "not(" + l() + ")" : 1 === u ? "(" + l() + " || " + l() + ")" : 2 === u ? i.pop() + " " + i.pop() : a(u))
      }
      return l();

      function l() {
         return i.pop().replace(/^\((.+)\)$/, "$1")
      }
   }, o = (e, r) => {
      if (0 in e) {
         r = t(r);
         var n = e[0],
            a = n < 0;
         a && (n = -n - 1);
         for (var i = 0, u = 1, l = !0;; u++, i++) {
            var s, f, d = u < e.length ? (typeof e[u])[0] : "";
            if (i >= r.length || "o" == (f = (typeof (s = r[i]))[0])) return !l || ("u" == d ? u > n && !a : "" == d != a);
            if ("u" == f) {
               if (!l || "u" != d) return !1
            } else if (l)
               if (d == f)
                  if (u <= n) {
                     if (s != e[u]) return !1
                  } else {
                     if (a ? s > e[u] : s < e[u]) return !1;
                     s != e[u] && (l = !1)
                  }
            else if ("s" != d && "n" != d) {
               if (a || u <= n) return !1;
               l = !1, u--
            } else {
               if (u <= n || f < d != a) return !1;
               l = !1
            } else "s" != d && "n" != d && (l = !1, u--)
         }
      }
      var c = [],
         p = c.pop.bind(c);
      for (i = 1; i < e.length; i++) {
         var h = e[i];
         c.push(1 == h ? p() | p() : 2 == h ? p() & p() : h ? o(h, r) : !p())
      }
      return !!p()
   }, i = (e, r) => {
      var t = g.S[e];
      if (!t || !g.o(t, r)) throw new Error("Shared module " + r + " doesn't exist in shared scope " + e);
      return t
   }, u = (e, r) => {
      var t = e[r];
      return Object.keys(t).reduce(((e, r) => !e || !t[e].loaded && n(e, r) ? r : e), 0)
   }, l = (e, r, t, n) => "Unsatisfied version " + t + " from " + (t && e[r][t].from) + " of shared singleton module " + r + " (required " + a(n) + ")", s = (e, r, t, n) => {
      var a = u(e, t);
      return o(n, a) || "undefined" != typeof console && console.warn && console.warn(l(e, t, a, n)), f(e[t][a])
   }, f = e => (e.loaded = 1, e.get()), d = (e => function (r, t, n, a) {
      var o = g.I(r);
      return o && o.then ? o.then(e.bind(e, r, g.S[r], t, n, a)) : e(r, g.S[r], t, n)
   })(((e, r, t, n) => (i(e, t), s(r, 0, t, n)))), c = {}, p = {
      643: () => d("default", "@jupyterlab/apputils", [1, 4, 0, 0, , "alpha", 11])
   }, h = {
      568: [643]
   }, g.f.consumes = (e, r) => {
      g.o(h, e) && h[e].forEach((e => {
         if (g.o(c, e)) return r.push(c[e]);
         var t = r => {
               c[e] = 0, g.m[e] = t => {
                  delete g.c[e], t.exports = r()
               }
            },
            n = r => {
               delete c[e], g.m[e] = t => {
                  throw delete g.c[e], r
               }
            };
         try {
            var a = p[e]();
            a.then ? r.push(c[e] = a.then(t).catch(n)) : t(a)
         } catch (e) {
            n(e)
         }
      }))
   }, (() => {
      var e = {
         121: 0
      };
      g.f.j = (r, t) => {
         var n = g.o(e, r) ? e[r] : void 0;
         if (0 !== n)
            if (n) t.push(n[2]);
            else {
               var a = new Promise(((t, a) => n = e[r] = [t, a]));
               t.push(n[2] = a);
               var o = g.p + g.u(r),
                  i = new Error;
               g.l(o, (t => {
                  if (g.o(e, r) && (0 !== (n = e[r]) && (e[r] = void 0), n)) {
                     var a = t && ("load" === t.type ? "missing" : t.type),
                        o = t && t.target && t.target.src;
                     i.message = "Loading chunk " + r + " failed.\n(" + a + ": " + o + ")", i.name = "ChunkLoadError", i.type = a, i.request = o, n[1](i)
                  }
               }), "chunk-" + r, r)
            }
      };
      var r = (r, t) => {
            var n, a, [o, i, u] = t,
               l = 0;
            if (o.some((r => 0 !== e[r]))) {
               for (n in i) g.o(i, n) && (g.m[n] = i[n]);
               u && u(g)
            }
            for (r && r(t); l < o.length; l++) a = o[l], g.o(e, a) && e[a] && e[a][0](), e[a] = 0
         },
         t = self.webpackChunk_superduperdb_theme_darcula = self.webpackChunk_superduperdb_theme_darcula || [];
      t.forEach(r.bind(null, 0)), t.push = r.bind(null, t.push.bind(t))
   })();
   var b = g(254);
   (_JUPYTERLAB = void 0 === _JUPYTERLAB ? {} : _JUPYTERLAB)["@superduperdb/theme-darcula"] = b
})();