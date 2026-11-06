/** Icon nét vẽ cho từng hiệu ứng (SVG tĩnh, không chứa dữ liệu người dùng). */
const ICONS: Record<string, string> = {
  pencil: '<path d="M4 20l4-1L19 8l-3-3L5 16z"/><path d="M14 7l3 3"/>',
  sketch: '<path d="M3 17c3-6 6-9 9-9s4 3 9 1"/><path d="M5 21l3-3M9 21l3-3M13 21l3-3"/>',
  lineart: '<path d="M4 18c2-8 6-12 9-12 4 0 3 6 7 6"/><circle cx="8" cy="9" r="2"/>',
  cartoon: '<circle cx="12" cy="12" r="8"/><circle cx="9" cy="10" r="1"/><circle cx="15" cy="10" r="1"/><path d="M8.5 14.5c2 2 5 2 7 0"/>',
  color_pencil: '<path d="M4 20l4-1L19 8l-3-3L5 16z"/><path d="M3 21h6" stroke-dasharray="2 2"/>',
  oil: '<path d="M14 4l6 6-8 8H6v-6z"/><path d="M6 18c-1 2-2 3-3 3 0-2 1-3 3-3z"/>',
};

export function effectIcon(id: string): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = ICONS[id] ?? ICONS.pencil;
  return svg;
}
