export const WASTE_CLASSES = [
  'Cardboard',
  'Food Organics',
  'Glass',
  'Metal',
  'Miscellaneous Trash',
  'Paper',
  'Plastic',
  'Textile Trash',
  'Vegetation'
];

export function getWasteLabel(index: number): string {
  if (index >= 0 && index < WASTE_CLASSES.length) {
    return WASTE_CLASSES[index];
  }
  return `Unknown Class (${index})`;
}

export function getAllWasteLabels(): string[] {
  return [...WASTE_CLASSES];
}
