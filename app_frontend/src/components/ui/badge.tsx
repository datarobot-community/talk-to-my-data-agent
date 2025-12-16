import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';

import { cn } from '@/lib/utils';

const BADGE_TYPE = {
  default: 'default',
  outline: 'outline',
} as const;

const BADGE_VARIANT = {
  default: 'default',
  destructive: 'destructive',
  info: 'info',
  warning: 'warning',
  success: 'success',
} as const;

// Hover styles map - only applied when onClick is provided
const BADGE_HOVER_STYLES: Record<
  keyof typeof BADGE_TYPE,
  Record<keyof typeof BADGE_VARIANT, string>
> = {
  [BADGE_TYPE.default]: {
    [BADGE_VARIANT.destructive]:
      'hover:bg-[color-mix(in_srgb,var(--destructive)_90%,black)] dark:hover:bg-[color-mix(in_srgb,var(--destructive)_80%,white)]',
    [BADGE_VARIANT.info]:
      'hover:bg-[color-mix(in_srgb,var(--link)_90%,black)] dark:hover:bg-[color-mix(in_srgb,var(--link)_80%,white)]',
    [BADGE_VARIANT.warning]:
      'hover:bg-[color-mix(in_srgb,var(--warning)_90%,black)] dark:hover:bg-[color-mix(in_srgb,var(--warning)_80%,white)]',
    [BADGE_VARIANT.success]:
      'hover:bg-[color-mix(in_srgb,var(--success)_90%,black)] dark:hover:bg-[color-mix(in_srgb,var(--success)_80%,white)]',
    [BADGE_VARIANT.default]:
      'hover:bg-[color-mix(in_srgb,var(--border)_90%,black)] dark:hover:bg-[color-mix(in_srgb,var(--border)_80%,white)]',
  },
  [BADGE_TYPE.outline]: {
    [BADGE_VARIANT.destructive]:
      'hover:border-[color-mix(in_srgb,var(--destructive)_75%,black)] hover:text-[color-mix(in_srgb,var(--destructive)_75%,black)] dark:hover:border-[color-mix(in_srgb,var(--destructive)_80%,white)] dark:hover:text-[color-mix(in_srgb,var(--destructive)_80%,white)]',
    [BADGE_VARIANT.info]:
      'hover:border-[color-mix(in_srgb,var(--link)_75%,black)] hover:text-[color-mix(in_srgb,var(--link)_75%,black)] dark:hover:border-[color-mix(in_srgb,var(--link)_80%,white)] dark:hover:text-[color-mix(in_srgb,var(--link)_80%,white)]',
    [BADGE_VARIANT.warning]:
      'hover:border-[color-mix(in_srgb,var(--warning)_75%,black)] hover:text-[color-mix(in_srgb,var(--warning)_75%,black)] dark:hover:border-[color-mix(in_srgb,var(--warning)_80%,white)] dark:hover:text-[color-mix(in_srgb,var(--warning)_80%,white)]',
    [BADGE_VARIANT.success]:
      'hover:border-[color-mix(in_srgb,var(--success)_75%,black)] hover:text-[color-mix(in_srgb,var(--success)_75%,black)] dark:hover:border-[color-mix(in_srgb,var(--success)_80%,white)] dark:hover:text-[color-mix(in_srgb,var(--success)_80%,white)]',
    [BADGE_VARIANT.default]:
      'hover:border-[color-mix(in_srgb,var(--secondary-foreground)_75%,black)] hover:text-[color-mix(in_srgb,var(--secondary-foreground)_75%,black)] dark:hover:border-[color-mix(in_srgb,var(--secondary-foreground)_80%,white)] dark:hover:text-[color-mix(in_srgb,var(--secondary-foreground)_80%,white)]',
  },
};

const BADGE_VARIANTS = cva(
  'inline-flex items-center rounded-xl border px-2 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
  {
    variants: {
      type: {
        [BADGE_TYPE.default]: '',
        [BADGE_TYPE.outline]: '',
      },
      variant: {
        [BADGE_VARIANT.default]: '',
        [BADGE_VARIANT.destructive]: '',
        [BADGE_VARIANT.info]: '',
        [BADGE_VARIANT.warning]: '',
        [BADGE_VARIANT.success]: '',
      },
    },
    compoundVariants: [
      // Default type - destructive (error)
      {
        type: BADGE_TYPE.default,
        variant: BADGE_VARIANT.destructive,
        className:
          'border-transparent bg-[var(--destructive)] text-[var(--primary-foreground)] shadow',
      },
      // Default type - info
      {
        type: BADGE_TYPE.default,
        variant: BADGE_VARIANT.info,
        className: 'border-transparent bg-[var(--link)] text-[var(--primary-foreground)] shadow',
      },
      // Default type - warning
      {
        type: BADGE_TYPE.default,
        variant: BADGE_VARIANT.warning,
        className: 'border-transparent bg-[var(--warning)] text-[var(--primary-foreground)] shadow',
      },
      // Default type - success
      {
        type: BADGE_TYPE.default,
        variant: BADGE_VARIANT.success,
        className: 'border-transparent bg-[var(--success)] text-[var(--primary-foreground)] shadow',
      },
      // Default type - default
      {
        type: BADGE_TYPE.default,
        variant: BADGE_VARIANT.default,
        className:
          'border-transparent bg-[var(--border)] text-[var(--foreground)] shadow dark:bg-[var(--border)]',
      },
      // Outline type - destructive (error)
      {
        type: BADGE_TYPE.outline,
        variant: BADGE_VARIANT.destructive,
        className:
          'border-[var(--destructive)] text-[var(--destructive)] bg-transparent dark:border-[color-mix(in_srgb,var(--destructive)_70%,white)] dark:text-[color-mix(in_srgb,var(--destructive)_70%,white)]',
      },
      // Outline type - info
      {
        type: BADGE_TYPE.outline,
        variant: BADGE_VARIANT.info,
        className: 'border-[var(--link)] text-[var(--link)] bg-transparent',
      },
      // Outline type - warning
      {
        type: BADGE_TYPE.outline,
        variant: BADGE_VARIANT.warning,
        className: 'border-[var(--warning)] text-[var(--warning)] bg-transparent',
      },
      // Outline type - success
      {
        type: BADGE_TYPE.outline,
        variant: BADGE_VARIANT.success,
        className: 'border-[var(--success)] text-[var(--success)] bg-transparent',
      },
      // Outline type - default
      {
        type: BADGE_TYPE.outline,
        variant: BADGE_VARIANT.default,
        className:
          'border-[var(--secondary-foreground)] text-[var(--secondary-foreground)] bg-transparent',
      },
    ],
    defaultVariants: {
      type: BADGE_TYPE.default,
      variant: BADGE_VARIANT.default,
    },
  }
);

const Badge = React.forwardRef<
  HTMLDivElement | HTMLButtonElement,
  React.HTMLAttributes<HTMLDivElement> &
    VariantProps<typeof BADGE_VARIANTS> & {
      type?: keyof typeof BADGE_TYPE;
      variant?: keyof typeof BADGE_VARIANT;
      testId?: string;
    }
>(({ className, type, variant, onClick, testId, ...props }, ref) => {
  const badgeType = type ?? BADGE_TYPE.default;
  const badgeVariant = variant ?? BADGE_VARIANT.default;
  const hasOnClick = !!onClick;

  const badgeClassName = cn(
    BADGE_VARIANTS({ type: badgeType, variant: badgeVariant }),
    hasOnClick && 'cursor-pointer',
    hasOnClick && BADGE_HOVER_STYLES[badgeType]?.[badgeVariant],
    className
  );

  if (hasOnClick) {
    return (
      <button
        data-testid={testId}
        ref={ref as React.ForwardedRef<HTMLButtonElement>}
        type="button"
        data-slot="badge"
        onClick={onClick as unknown as React.MouseEventHandler<HTMLButtonElement>}
        className={badgeClassName}
        {...(props as React.ButtonHTMLAttributes<HTMLButtonElement>)}
      />
    );
  }

  return (
    <div
      data-testid={testId}
      ref={ref as React.ForwardedRef<HTMLDivElement>}
      data-slot="badge"
      className={badgeClassName}
      {...props}
    />
  );
});

Badge.displayName = 'Badge';

export { Badge, BADGE_VARIANTS, BADGE_TYPE, BADGE_VARIANT };
