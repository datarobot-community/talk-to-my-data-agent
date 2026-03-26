import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';

import { cn } from '@/lib/utils';

const BUTTON_VARIANT = {
  primary: 'primary',
  secondary: 'secondary',
  destructive: 'destructive',
  ghost: 'ghost',
  link: 'link',
} as const;

const BUTTON_SIZE = {
  default: 'default',
  sm: 'sm',
  lg: 'lg',
  icon: 'icon',
  iconSm: 'icon-sm',
} as const;

const BUTTON_VARIANTS = cva(
  "inline-flex shrink-0 cursor-pointer items-center justify-center gap-1 rounded text-sm font-semibold whitespace-nowrap transition-all outline-none focus-visible:border-ring focus-visible:ring-[1px] focus-visible:ring-ring disabled:pointer-events-none aria-invalid:border-destructive aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  {
    variants: {
      size: {
        [BUTTON_SIZE.default]: 'h-9 px-4 py-2 has-[>svg]:px-3',
        [BUTTON_SIZE.sm]: 'h-8 gap-1.5 px-3 has-[>svg]:px-2.5',
        [BUTTON_SIZE.lg]: 'h-10 px-6 has-[>svg]:px-4',
        [BUTTON_SIZE.icon]: 'size-9',
        [BUTTON_SIZE.iconSm]: 'size-5',
      },
      variant: {
        [BUTTON_VARIANT.primary]:
          'bg-primary text-primary-foreground hover:bg-accent disabled:bg-muted disabled:text-muted-foreground',
        [BUTTON_VARIANT.destructive]:
          'bg-destructive text-white hover:bg-destructive/90 disabled:brightness-70',
        [BUTTON_VARIANT.secondary]:
          'border border-muted-foreground bg-transparent hover:border-accent hover:bg-muted/50 hover:text-accent disabled:border-foreground/50 disabled:text-foreground/50',
        [BUTTON_VARIANT.ghost]:
          'px-2 hover:bg-sidebar-primary hover:text-accent disabled:text-foreground/50',
        [BUTTON_VARIANT.link]: 'p-0 text-primary hover:text-accent disabled:text-foreground/50',
      },
    },
    defaultVariants: {
      variant: BUTTON_VARIANT.primary,
      size: BUTTON_SIZE.default,
    },
  }
);

const Button = React.forwardRef<
  HTMLButtonElement,
  React.ComponentProps<'button'> &
    VariantProps<typeof BUTTON_VARIANTS> & {
      asChild?: boolean;
      testId?: string;
    }
>(({ className, variant, size, asChild = false, testId, ...props }, ref) => {
  const Comp = asChild ? Slot : 'button';

  const buttonElement = (
    <Comp
      data-testid={testId}
      ref={ref}
      data-slot="button"
      className={cn(BUTTON_VARIANTS({ variant, size, className }))}
      {...props}
    />
  );

  /* wrap with span so the tooltip (title) will be shown even when disabled */
  if (props.disabled && props.title) {
    return <span title={props.title}>{buttonElement}</span>;
  }

  return buttonElement;
});

Button.displayName = 'Button';

export { Button, BUTTON_VARIANTS, BUTTON_VARIANT, BUTTON_SIZE };
