import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import jsxA11y from 'eslint-plugin-jsx-a11y';
import prettier from 'eslint-plugin-prettier';
import prettierConfig from 'eslint-config-prettier';
import tanstackQuery from '@tanstack/eslint-plugin-query';
import eslintPluginBetterTailwindcss from 'eslint-plugin-better-tailwindcss';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist'] },
  {
    extends: [
      js.configs.recommended,
      ...tseslint.configs.recommended,
      ...tanstackQuery.configs['flat/recommended'],
      jsxA11y.flatConfigs.recommended,
      prettierConfig,
    ],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
      '@tanstack/query': tanstackQuery,
      prettier: prettier,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
      'prettier/prettier': 'error',
      // All 5 autofocus sites move focus into a surface the user just opened (new-chat
      // modal, chat prompt, inline cell editor, search field) or pass the prop through.
      // Leaving the rule on would only produce five disable comments.
      'jsx-a11y/no-autofocus': 'off',
      'no-restricted-imports': [
        'error',
        {
          paths: [
            {
              name: 'react-i18next',
              importNames: ['useTranslation'],
              message: 'Import useTranslation from @/i18n instead of react-i18next directly',
            },
            {
              name: 'i18next',
              message: 'Import i18n from @/i18n instead of i18next directly',
            },
          ],
        },
      ],
    },
  },
  {
    files: ['**/*.{jsx,tsx}'],
    plugins: {
      'better-tailwindcss': eslintPluginBetterTailwindcss,
    },
    rules: {
      'better-tailwindcss/enforce-consistent-class-order': ['error', { order: 'official' }],
      'better-tailwindcss/enforce-shorthand-classes': 'error',
      'better-tailwindcss/no-conflicting-classes': 'error',
      'better-tailwindcss/no-duplicate-classes': 'error',
      'better-tailwindcss/no-unnecessary-whitespace': 'error',
      'better-tailwindcss/no-deprecated-classes': 'off',
      'better-tailwindcss/enforce-consistent-variable-syntax': ['error', { syntax: 'shorthand' }],
      'better-tailwindcss/enforce-consistent-important-position': [
        'error',
        { position: 'recommended' },
      ],
    },
    settings: {
      'better-tailwindcss': {
        entryPoint: 'src/index.css',
      },
    },
  },
  // Allow direct i18n imports in the i18n setup files
  {
    files: ['src/i18n/**/*.{ts,tsx}', 'src/main.tsx'],
    rules: {
      'no-restricted-imports': 'off',
    },
  },
  // Disable no-explicit-any for test files
  {
    files: ['**/*.test.{ts,tsx}'],
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
    },
  },
  {
    // Test fixtures deliberately attach handlers to bare elements to assert event
    // behaviour; they are not shipped UI. dr-ui and the design system relax the
    // interaction rules under their own test globs.
    files: ['tests/**/*.{ts,tsx}', '**/*.{test,spec}.{ts,tsx}'],
    rules: {
      'jsx-a11y/click-events-have-key-events': 'off',
      'jsx-a11y/no-static-element-interactions': 'off',
      'jsx-a11y/no-noninteractive-element-interactions': 'off',
    },
  }
);