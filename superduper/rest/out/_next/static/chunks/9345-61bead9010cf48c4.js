"use strict";(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[9345],{10269:function(e,r,t){t.d(r,{Z:function(){return n}});let n=(0,t(84313).Z)("X",[["path",{d:"M18 6 6 18",key:"1bl5f8"}],["path",{d:"m6 6 12 12",key:"d8bk6v"}]])},96764:function(e,r,t){var n=t(7653);let o=n.forwardRef(function(e,r){let{title:t,titleId:o,...a}=e;return n.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 20 20",fill:"currentColor","aria-hidden":"true","data-slot":"icon",ref:r,"aria-labelledby":o},a),t?n.createElement("title",{id:o},t):null,n.createElement("path",{fillRule:"evenodd",d:"M8.22 5.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.06-1.06L11.94 10 8.22 6.28a.75.75 0 0 1 0-1.06Z",clipRule:"evenodd"}))});r.Z=o},61284:function(e,r,t){t.d(r,{VY:function(){return L},ee:function(){return X},fC:function(){return S},h_:function(){return K},xp:function(){return z},xz:function(){return Y}});var n=t(7653),o=t(46196),a=t(94492),l=t(99933),u=t(97256),i=t(60307),s=t(42142),c=t(17321),d=t(51209),p=t(20153),f=t(65622),v=t(78378),h=t(8828),m=t(47178),x=t(27130),g=t(34674),C=t(27573),b="Popover",[w,P]=(0,l.b)(b,[d.D7]),j=(0,d.D7)(),[R,y]=w(b),N=e=>{let{__scopePopover:r,children:t,open:o,defaultOpen:a,onOpenChange:l,modal:u=!1}=e,i=j(r),s=n.useRef(null),[p,f]=n.useState(!1),[v=!1,h]=(0,m.T)({prop:o,defaultProp:a,onChange:l});return(0,C.jsx)(d.fC,{...i,children:(0,C.jsx)(R,{scope:r,contentId:(0,c.M)(),triggerRef:s,open:v,onOpenChange:h,onOpenToggle:n.useCallback(()=>h(e=>!e),[h]),hasCustomAnchor:p,onCustomAnchorAdd:n.useCallback(()=>f(!0),[]),onCustomAnchorRemove:n.useCallback(()=>f(!1),[]),modal:u,children:t})})};N.displayName=b;var _="PopoverAnchor",k=n.forwardRef((e,r)=>{let{__scopePopover:t,...o}=e,a=y(_,t),l=j(t),{onCustomAnchorAdd:u,onCustomAnchorRemove:i}=a;return n.useEffect(()=>(u(),()=>i()),[u,i]),(0,C.jsx)(d.ee,{...l,...o,ref:r})});k.displayName=_;var M="PopoverTrigger",O=n.forwardRef((e,r)=>{let{__scopePopover:t,...n}=e,l=y(M,t),u=j(t),i=(0,a.e)(r,l.triggerRef),s=(0,C.jsx)(v.WV.button,{type:"button","aria-haspopup":"dialog","aria-expanded":l.open,"aria-controls":l.contentId,"data-state":B(l.open),...n,ref:i,onClick:(0,o.M)(e.onClick,l.onOpenToggle)});return l.hasCustomAnchor?s:(0,C.jsx)(d.ee,{asChild:!0,...u,children:s})});O.displayName=M;var E="PopoverPortal",[D,A]=w(E,{forceMount:void 0}),F=e=>{let{__scopePopover:r,forceMount:t,children:n,container:o}=e,a=y(E,r);return(0,C.jsx)(D,{scope:r,forceMount:t,children:(0,C.jsx)(f.z,{present:t||a.open,children:(0,C.jsx)(p.h,{asChild:!0,container:o,children:n})})})};F.displayName=E;var I="PopoverContent",V=n.forwardRef((e,r)=>{let t=A(I,e.__scopePopover),{forceMount:n=t.forceMount,...o}=e,a=y(I,e.__scopePopover);return(0,C.jsx)(f.z,{present:n||a.open,children:a.modal?(0,C.jsx)(Z,{...o,ref:r}):(0,C.jsx)($,{...o,ref:r})})});V.displayName=I;var Z=n.forwardRef((e,r)=>{let t=y(I,e.__scopePopover),l=n.useRef(null),u=(0,a.e)(r,l),i=n.useRef(!1);return n.useEffect(()=>{let e=l.current;if(e)return(0,x.Ry)(e)},[]),(0,C.jsx)(g.Z,{as:h.g7,allowPinchZoom:!0,children:(0,C.jsx)(T,{...e,ref:u,trapFocus:t.open,disableOutsidePointerEvents:!0,onCloseAutoFocus:(0,o.M)(e.onCloseAutoFocus,e=>{var r;e.preventDefault(),i.current||null===(r=t.triggerRef.current)||void 0===r||r.focus()}),onPointerDownOutside:(0,o.M)(e.onPointerDownOutside,e=>{let r=e.detail.originalEvent,t=0===r.button&&!0===r.ctrlKey,n=2===r.button||t;i.current=n},{checkForDefaultPrevented:!1}),onFocusOutside:(0,o.M)(e.onFocusOutside,e=>e.preventDefault(),{checkForDefaultPrevented:!1})})})}),$=n.forwardRef((e,r)=>{let t=y(I,e.__scopePopover),o=n.useRef(!1),a=n.useRef(!1);return(0,C.jsx)(T,{...e,ref:r,trapFocus:!1,disableOutsidePointerEvents:!1,onCloseAutoFocus:r=>{var n,l;null===(n=e.onCloseAutoFocus)||void 0===n||n.call(e,r),r.defaultPrevented||(o.current||null===(l=t.triggerRef.current)||void 0===l||l.focus(),r.preventDefault()),o.current=!1,a.current=!1},onInteractOutside:r=>{var n,l;null===(n=e.onInteractOutside)||void 0===n||n.call(e,r),r.defaultPrevented||(o.current=!0,"pointerdown"!==r.detail.originalEvent.type||(a.current=!0));let u=r.target;(null===(l=t.triggerRef.current)||void 0===l?void 0:l.contains(u))&&r.preventDefault(),"focusin"===r.detail.originalEvent.type&&a.current&&r.preventDefault()}})}),T=n.forwardRef((e,r)=>{let{__scopePopover:t,trapFocus:n,onOpenAutoFocus:o,onCloseAutoFocus:a,disableOutsidePointerEvents:l,onEscapeKeyDown:c,onPointerDownOutside:p,onFocusOutside:f,onInteractOutside:v,...h}=e,m=y(I,t),x=j(t);return(0,i.EW)(),(0,C.jsx)(s.M,{asChild:!0,loop:!0,trapped:n,onMountAutoFocus:o,onUnmountAutoFocus:a,children:(0,C.jsx)(u.XB,{asChild:!0,disableOutsidePointerEvents:l,onInteractOutside:v,onEscapeKeyDown:c,onPointerDownOutside:p,onFocusOutside:f,onDismiss:()=>m.onOpenChange(!1),children:(0,C.jsx)(d.VY,{"data-state":B(m.open),role:"dialog",id:m.contentId,...x,...h,ref:r,style:{...h.style,"--radix-popover-content-transform-origin":"var(--radix-popper-transform-origin)","--radix-popover-content-available-width":"var(--radix-popper-available-width)","--radix-popover-content-available-height":"var(--radix-popper-available-height)","--radix-popover-trigger-width":"var(--radix-popper-anchor-width)","--radix-popover-trigger-height":"var(--radix-popper-anchor-height)"}})})})}),W="PopoverClose",z=n.forwardRef((e,r)=>{let{__scopePopover:t,...n}=e,a=y(W,t);return(0,C.jsx)(v.WV.button,{type:"button",...n,ref:r,onClick:(0,o.M)(e.onClick,()=>a.onOpenChange(!1))})});function B(e){return e?"open":"closed"}z.displayName=W,n.forwardRef((e,r)=>{let{__scopePopover:t,...n}=e,o=j(t);return(0,C.jsx)(d.Eh,{...o,...n,ref:r})}).displayName="PopoverArrow";var S=N,X=k,Y=O,K=F,L=V},2177:function(e,r,t){t.d(r,{z$:function(){return b},fC:function(){return C}});var n=t(7653),o=t(27573),a=t(78378),l="Progress",[u,i]=function(e,r=[]){let t=[],a=()=>{let r=t.map(e=>n.createContext(e));return function(t){let o=t?.[e]||r;return n.useMemo(()=>({[`__scope${e}`]:{...t,[e]:o}}),[t,o])}};return a.scopeName=e,[function(r,a){let l=n.createContext(a),u=t.length;function i(r){let{scope:t,children:a,...i}=r,s=t?.[e][u]||l,c=n.useMemo(()=>i,Object.values(i));return(0,o.jsx)(s.Provider,{value:c,children:a})}return t=[...t,a],i.displayName=r+"Provider",[i,function(t,o){let i=o?.[e][u]||l,s=n.useContext(i);if(s)return s;if(void 0!==a)return a;throw Error(`\`${t}\` must be used within \`${r}\``)}]},function(...e){let r=e[0];if(1===e.length)return r;let t=()=>{let t=e.map(e=>({useScope:e(),scopeName:e.scopeName}));return function(e){let o=t.reduce((r,{useScope:t,scopeName:n})=>{let o=t(e)[`__scope${n}`];return{...r,...o}},{});return n.useMemo(()=>({[`__scope${r.scopeName}`]:o}),[o])}};return t.scopeName=r.scopeName,t}(a,...r)]}(l),[s,c]=u(l),d=n.forwardRef((e,r)=>{var t,n,l,u;let{__scopeProgress:i,value:c=null,max:d,getValueLabel:p=v,...f}=e;(d||0===d)&&!x(d)&&console.error((t="".concat(d),n="Progress","Invalid prop `max` of value `".concat(t,"` supplied to `").concat(n,"`. Only numbers greater than 0 are valid max values. Defaulting to `").concat(100,"`.")));let C=x(d)?d:100;null===c||g(c,C)||console.error((l="".concat(c),u="Progress","Invalid prop `value` of value `".concat(l,"` supplied to `").concat(u,"`. The `value` prop must be:\n  - a positive number\n  - less than the value passed to `max` (or ").concat(100," if no `max` prop is set)\n  - `null` or `undefined` if the progress is indeterminate.\n\nDefaulting to `null`.")));let b=g(c,C)?c:null,w=m(b)?p(b,C):void 0;return(0,o.jsx)(s,{scope:i,value:b,max:C,children:(0,o.jsx)(a.WV.div,{"aria-valuemax":C,"aria-valuemin":0,"aria-valuenow":m(b)?b:void 0,"aria-valuetext":w,role:"progressbar","data-state":h(b,C),"data-value":null!=b?b:void 0,"data-max":C,...f,ref:r})})});d.displayName=l;var p="ProgressIndicator",f=n.forwardRef((e,r)=>{var t;let{__scopeProgress:n,...l}=e,u=c(p,n);return(0,o.jsx)(a.WV.div,{"data-state":h(u.value,u.max),"data-value":null!==(t=u.value)&&void 0!==t?t:void 0,"data-max":u.max,...l,ref:r})});function v(e,r){return"".concat(Math.round(e/r*100),"%")}function h(e,r){return null==e?"indeterminate":e===r?"complete":"loading"}function m(e){return"number"==typeof e}function x(e){return m(e)&&!isNaN(e)&&e>0}function g(e,r){return m(e)&&!isNaN(e)&&e<=r&&e>=0}f.displayName=p;var C=d,b=f}}]);