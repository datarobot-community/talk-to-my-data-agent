import type { Config } from 'tailwindcss';
import defaultTheme from 'tailwindcss/defaultTheme';
import cssVariablesPlugin, { drColors } from './css-variables';
import componentsClassesPlugin from './components-classes';

const config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './registry/**/*.{js,ts,jsx,tsx,mdx}',
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        border: 'var(--border)',
        input: 'var(--input)',
        ring: 'var(--ring)',
        background: 'var(--background)',
        foreground: 'var(--foreground)',
        primary: {
          DEFAULT: 'var(--primary)',
          foreground: 'var(--primary-foreground)',
        },
        secondary: {
          DEFAULT: 'var(--secondary)',
          foreground: 'var(--secondary-foreground)',
        },
        destructive: {
          DEFAULT: 'var(--destructive)',
          foreground: 'var(--destructive-foreground)',
        },
        muted: {
          DEFAULT: 'var(--muted)',
          foreground: 'var(--muted-foreground)',
        },
        accent: {
          DEFAULT: 'var(--accent)',
          foreground: 'var(--accent-foreground)',
        },
        popover: {
          DEFAULT: 'var(--popover)',
          foreground: 'var(--popover-foreground)',
        },
        card: {
          DEFAULT: 'var(--card)',
          foreground: 'var(--card-foreground)',
        },
        sidebar: {
          DEFAULT: 'var(--sidebar-background)',
          foreground: 'var(--sidebar-foreground)',
          primary: 'var(--sidebar-primary)',
          'primary-foreground': 'var(--sidebar-primary-foreground)',
          accent: 'var(--sidebar-accent)',
          'accent-foreground': 'var(--sidebar-accent-foreground)',
          border: 'var(--sidebar-border)',
          ring: 'var(--sidebar-ring)',
        },
        link: 'var(--link)',
        success: 'var(--success)',
        warning: 'var(--warning)',
        ...drColors,
      },
      borderRadius: {
        md: '0.25rem',
        DEFAULT: '0.375rem',
      },
      fontSize: {
        xl: ['1.25rem', { lineHeight: '1.5rem' }],
        '2xl': ['1.5rem', { lineHeight: '1.75rem' }],
        '3xl': ['1.75rem', { lineHeight: '2.25rem' }],
        '4xl': ['2.5rem', { lineHeight: '3rem' }],
      },
      fontFamily: {
        serif: ['var(--font-inter-serif)', ...defaultTheme.fontFamily.serif],
        mono: ['var(--font-roboto-mono)', ...defaultTheme.fontFamily.mono],
      },
    },
  },
  plugins: [cssVariablesPlugin, componentsClassesPlugin],
} satisfies Config;

export default config;
