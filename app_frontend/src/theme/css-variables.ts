import type { PluginCreator } from 'tailwindcss/plugin';
import colors from './colors.json';
import semanticTokens from './semantic-tokens.json';

const drColors = Object.values(colors).reduce((acc, value) => ({ ...acc, ...value }), {});

const paletteVars = Object.entries(drColors).reduce<Record<string, string>>(
  (acc, [colorName, colorValue]) => {
    acc[`--${colorName}`] = colorValue as string;
    return acc;
  },
  {}
);

const lightVars: Record<string, string> = {};
const darkVars: Record<string, string> = {};
for (const group of semanticTokens.groups) {
  for (const token of group.tokens) {
    lightVars[`--${token.name}`] = `var(--${token.light})`;
    darkVars[`--${token.name}`] = `var(--${token.dark})`;
  }
}

const cssVariablesPlugin: PluginCreator = function cssVariablesPlugin({ addBase }) {
  addBase({
    ':root': { ...paletteVars, ...lightVars },
    '.dark': darkVars,
  });
};
export { drColors };
export default cssVariablesPlugin;
