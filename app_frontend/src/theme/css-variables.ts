import type { PluginCreator } from 'tailwindcss/plugin';
import colors from './colors.json';

const drColors = Object.values(colors).reduce((acc, value) => ({ ...acc, ...value }), {});

const cssVariablesPlugin: PluginCreator = function cssVariablesPlugin({ addBase }) {
  const cssVariables = Object.entries(drColors).reduce((acc, [colorName, colorValue]) => {
    return {
      ...acc,
      [`--${colorName}`]: colorValue,
    };
  }, {});

  addBase({
    ':root': cssVariables,
  });
};
export { drColors };
export default cssVariablesPlugin;
