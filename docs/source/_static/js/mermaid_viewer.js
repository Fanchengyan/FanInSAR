(function () {
    'use strict';

    var MIN_SCALE = 0.05;
    var MAX_SCALE = 8;
    var states = new WeakMap();

    function clamp(value, lo, hi) {
        return Math.min(hi, Math.max(lo, value));
    }

    function naturalSize(svg) {
        var vb = svg.viewBox && svg.viewBox.baseVal;
        if (vb && vb.width > 0 && vb.height > 0) {
            return { w: vb.width, h: vb.height };
        }
        var rect = svg.getBoundingClientRect();
        return { w: rect.width || 800, h: rect.height || 600 };
    }

    function applyTransform(viewer, state, instant) {
        state.svg.style.transition = instant ? 'none' : 'transform 0.15s ease';
        state.svg.style.transform =
            'translate(' + state.tx + 'px, ' + state.ty + 'px) scale(' + state.k + ')';
    }

    function fitToWidth(viewer, state, instant) {
        var cw = viewer.clientWidth;
        var ch = viewer.clientHeight;
        if (!cw || !ch) return;
        state.k = clamp(cw / state.nw, MIN_SCALE, MAX_SCALE);
        state.tx = 0;
        state.ty = Math.max(0, (ch - state.nh * state.k) / 2);
        applyTransform(viewer, state, instant);
    }

    // Keep the point under the cursor fixed while the scale changes.
    function zoomAt(viewer, state, factor, cx, cy) {
        var next = clamp(state.k * factor, MIN_SCALE, MAX_SCALE);
        if (next === state.k) return;
        state.tx = cx - (cx - state.tx) * (next / state.k);
        state.ty = cy - (cy - state.ty) * (next / state.k);
        state.k = next;
        applyTransform(viewer, state, true);
    }

    function enableViewer(svg) {
        if (svg.getAttribute('data-mermaid-viewer') === 'true') return;
        if (!svg.closest('.mermaid-container-fullscreen')) return;
        svg.setAttribute('data-mermaid-viewer', 'true');

        var size = naturalSize(svg);
        svg.removeAttribute('style');
        svg.setAttribute('width', size.w);
        svg.setAttribute('height', size.h);
        svg.style.maxWidth = 'none';
        svg.style.width = size.w + 'px';
        svg.style.height = size.h + 'px';

        var viewer = document.createElement('div');
        viewer.className = 'mermaid-viewer';

        var hint = document.createElement('div');
        hint.className = 'mermaid-viewer-hint';
        hint.textContent = 'Drag to pan · Scroll to zoom · Double-click to fit';
        viewer.appendChild(hint);

        var toolbar = document.createElement('div');
        toolbar.className = 'mermaid-viewer-toolbar';
        toolbar.setAttribute('role', 'toolbar');
        viewer.appendChild(toolbar);

        var state = { svg: svg, nw: size.w, nh: size.h, k: 1, tx: 0, ty: 0 };
        states.set(viewer, state);

        var buttons = [
            { label: '−', title: 'Zoom out', action: function () { zoomCentered(1 / 1.25); } },
            { label: '+', title: 'Zoom in', action: function () { zoomCentered(1.25); } },
            { label: '⤢', title: 'Fit to width', action: function () { fitToWidth(viewer, state, false); } }
        ];
        buttons.forEach(function (button) {
            var el = document.createElement('button');
            el.type = 'button';
            el.textContent = button.label;
            el.title = button.title;
            el.setAttribute('aria-label', button.title);
            el.addEventListener('click', button.action);
            toolbar.appendChild(el);
        });

        function zoomCentered(factor) {
            zoomAt(viewer, state, factor, viewer.clientWidth / 2, viewer.clientHeight / 2);
        }

        var host = svg.closest('pre.mermaid') || svg.parentNode;
        host.parentNode.replaceChild(viewer, host);
        viewer.appendChild(svg);

        fitToWidth(viewer, state, true);
        requestAnimationFrame(function () { fitToWidth(viewer, state, true); });

        setTimeout(function () { hint.classList.add('show'); }, 600);
        setTimeout(function () { hint.classList.remove('show'); }, 3400);

        var drag = null;
        viewer.addEventListener('pointerdown', function (event) {
            if (event.target.closest('.mermaid-viewer-toolbar')) return;
            drag = { x: event.clientX, y: event.clientY, tx: state.tx, ty: state.ty };
            viewer.setPointerCapture(event.pointerId);
            viewer.classList.add('dragging');
            event.preventDefault();
        });
        viewer.addEventListener('pointermove', function (event) {
            if (!drag) return;
            state.tx = drag.tx + (event.clientX - drag.x);
            state.ty = drag.ty + (event.clientY - drag.y);
            applyTransform(viewer, state, true);
        });
        ['pointerup', 'pointercancel'].forEach(function (name) {
            viewer.addEventListener(name, function () {
                drag = null;
                viewer.classList.remove('dragging');
            });
        });

        viewer.addEventListener('wheel', function (event) {
            var rect = viewer.getBoundingClientRect();
            zoomAt(
                viewer,
                state,
                Math.exp(-event.deltaY * 0.0015),
                event.clientX - rect.left,
                event.clientY - rect.top
            );
            event.preventDefault();
        }, { passive: false });

        viewer.addEventListener('dblclick', function (event) {
            if (event.target.closest('.mermaid-viewer-toolbar')) return;
            fitToWidth(viewer, state, false);
        });
    }

    function enableClickToOpen(container) {
        if (container.getAttribute('data-click-to-open') === 'true') return;
        container.setAttribute('data-click-to-open', 'true');
        container.addEventListener('click', function (event) {
            if (event.target.closest('.mermaid-fullscreen-btn')) return;
            if (event.target.closest('.mermaid-fullscreen-modal')) return;
            var btn = container.querySelector('.mermaid-fullscreen-btn');
            if (btn) btn.click();
        });
    }

    var observer = new MutationObserver(function () {
        var svgs = document.querySelectorAll('.mermaid-container-fullscreen pre.mermaid svg');
        for (var i = 0; i < svgs.length; i++) enableViewer(svgs[i]);
        var containers = document.querySelectorAll('.mermaid-container');
        for (var j = 0; j < containers.length; j++) enableClickToOpen(containers[j]);
    });
    observer.observe(document.documentElement, { childList: true, subtree: true });

    document.addEventListener('keydown', function (event) {
        var modal = document.querySelector('.mermaid-fullscreen-modal.active');
        if (!modal) return;
        var viewer = modal.querySelector('.mermaid-viewer');
        var state = viewer && states.get(viewer);
        if (!viewer || !state) return;
        if (event.key === '+' || event.key === '=') {
            zoomCenteredKey(1.25);
            event.preventDefault();
        } else if (event.key === '-' || event.key === '_') {
            zoomCenteredKey(1 / 1.25);
            event.preventDefault();
        }
        function zoomCenteredKey(factor) {
            zoomAt(viewer, state, factor, viewer.clientWidth / 2, viewer.clientHeight / 2);
        }
    });
})();
