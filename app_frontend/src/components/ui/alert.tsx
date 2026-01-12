import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';

import { cn } from '@/lib/utils';

const ALERT_VARIANT = {
  destructive: 'destructive',
  info: 'info',
  warning: 'warning',
  success: 'success',
};

const alertVariants = cva(
  `
    relative w-full rounded-lg border bg-popover px-4 pt-4 pb-2 text-sm
    [&>svg]:absolute [&>svg]:top-4 [&>svg]:left-4 [&>svg]:h-[14] [&>svg]:w-[16]
    [&>svg+div]:translate-y-[-3px]
    [&>svg~*]:pl-6
  `,
  {
    variants: {
      variant: {
        [ALERT_VARIANT.info]: `
          border-primary
          [&>svg]:text-primary
        `,
        [ALERT_VARIANT.destructive]: `
          border-destructive/90
          [&>svg]:text-destructive/90
        `,
        [ALERT_VARIANT.warning]: `
          border-warning/75
          [&>svg]:text-warning/75
        `,
        [ALERT_VARIANT.success]: `
          border-success/75
          [&>svg]:text-success/75
        `,
      },
    },
    defaultVariants: {
      variant: ALERT_VARIANT.info,
    },
  }
);

function Alert({
  className,
  variant,
  ...props
}: React.ComponentProps<'div'> & VariantProps<typeof alertVariants>) {
  return (
    <div
      data-slot="alert"
      role="alert"
      className={cn(alertVariants({ variant }), className)}
      {...props}
    />
  );
}

function AlertTitle({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-slot="alert-title"
      className={cn('mb-1 font-normal text-primary', className)}
      {...props}
    />
  );
}

function AlertDescription({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      className={cn(
        `
          text-xs text-secondary-foreground
          [&_p]:leading-relaxed
        `,
        className
      )}
      {...props}
    />
  );
}

function AlertFooter({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-slot="alert-footer"
      className={cn(
        `
          mt-4 flex min-h-0 items-center gap-4 text-sm text-primary
          [&>*:first-child]:pl-0
          [&>a]:no-underline
        `,
        className
      )}
      {...props}
    />
  );
}

export { Alert, AlertTitle, AlertDescription, AlertFooter, ALERT_VARIANT };
