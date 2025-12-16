import type { PluginCreator } from 'tailwindcss/plugin';

const typography = {
  '.heading-01': textSetUp({
    fontWeight: '500',
    fontSize: '2.5rem',
    lineHeight: '3rem',
    letterSpacing: '-0.025rem',
  }),
  '.heading-02': textSetUp({
    fontWeight: '500',
    fontSize: '1.75rem',
    lineHeight: '2.25rem',
    letterSpacing: '-0.025rem',
  }),
  '.heading-03': textSetUp({
    fontWeight: '500',
    fontSize: '1.5rem',
    lineHeight: '1.75rem',
    letterSpacing: '-0.025rem',
  }),
  '.heading-04': textSetUp({
    fontWeight: '500',
    fontSize: '1.25rem',
    lineHeight: '1.5rem',
    letterSpacing: '-0.025rem',
  }),
  '.heading-05': textSetUp({
    fontWeight: '600',
    fontSize: '1rem',
    lineHeight: '1.25rem',
    letterSpacing: '-0.025rem',
  }),
  '.heading-06': textSetUp({
    fontWeight: '600',
    fontSize: '0.875rem',
    lineHeight: '1.25rem',
    letterSpacing: '-0.025rem',
  }),
  '.body': textSetUp({
    fontWeight: '400',
    fontSize: '0.875rem',
    lineHeight: '1.25rem',
  }),
  '.body-secondary': textSetUp({
    fontWeight: '400',
    fontSize: '0.875rem',
    lineHeight: '1.25rem',
    color: 'var(--color-muted-foreground)',
  }),
  '.uppercased': {
    ...textSetUp({
      fontWeight: '400',
      fontSize: '0.875rem',
      lineHeight: '1.25rem',
      color: 'var(--color-muted-foreground)',
      letterSpacing: '0.075rem',
    }),
    textTransform: 'uppercase',
  },
  '.mn-label': textSetUp({
    fontWeight: '600',
    fontSize: '0.875rem',
    lineHeight: '1.25rem',
    letterSpacing: '-0.025rem',
  }),
  '.mn-label-large': textSetUp({
    fontWeight: '600',
    fontSize: '1rem',
    lineHeight: '1.25rem',
    letterSpacing: '-0.025rem',
  }),
  '.caption-01': textSetUp({
    fontWeight: '400',
    fontSize: '0.75rem',
    lineHeight: '1rem',
    color: 'var(--color-muted-foreground)',
  }),
  '.code': textSetUp({
    fontWeight: '400',
    fontSize: '0.875rem',
    lineHeight: '1.25rem',
    fontFamily: 'var(--font-mono)',
  }),
  '.anchor': {
    ...textSetUp({
      fontWeight: '400',
      fontSize: '0.875rem',
      lineHeight: '1.25rem',
      color: 'var(--link)',
    }),
    textDecoration: 'none',
    borderBottom: '1px solid transparent',

    '&:hover': {
      borderColor: 'var(--link)',
      filter: 'brightness(1.1)',
    },
  },
  '.anchor-muted': {
    ...textSetUp({
      fontWeight: '500',
      fontSize: '0.875rem',
      lineHeight: '1.25rem',
      color: 'var(--foreground)',
    }),
    textDecoration: 'none',
    borderBottom: '1px solid transparent',

    '&:hover': {
      color: 'var(--link)',
      filter: 'brightness(1.1)',
      borderColor: 'var(--link)',
    },
  },
};

function textSetUp({
  fontWeight,
  fontSize,
  lineHeight = '1.5',
  color = 'var(--foreground)',
  fontFamily = 'var(--font-serif)',
  letterSpacing = 'normal',
}: {
  fontWeight: string;
  fontSize: string;
  lineHeight?: string;
  color?: string;
  fontFamily?: string;
  letterSpacing?: string;
}) {
  return {
    fontFamily: fontFamily,
    fontSize: fontSize,
    fontWeight: fontWeight,
    lineHeight: lineHeight,
    letterSpacing: letterSpacing,
    color: color,
    fill: color,
  };
}

const componentsClasses: PluginCreator = function componentsClasses({ addComponents }) {
  addComponents(typography);
};

export default componentsClasses;
