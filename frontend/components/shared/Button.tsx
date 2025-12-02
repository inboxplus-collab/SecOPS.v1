import * as React from "react";
import clsx from "clsx";

type Variant = "primary" | "secondary" | "ghost" | "danger";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  loading?: boolean;
}

const variantClasses: Record<Variant, string> = {
  primary:
    "bg-blue-600 text-white hover:bg-blue-500 disabled:bg-blue-900/40 disabled:text-blue-200",
  secondary:
    "bg-neutral-800 text-neutral-50 hover:bg-neutral-700 disabled:bg-neutral-900/40",
  ghost:
    "bg-transparent text-neutral-200 hover:bg-neutral-800/60 disabled:text-neutral-500",
  danger:
    "bg-red-600 text-white hover:bg-red-500 disabled:bg-red-900/40 disabled:text-red-200",
};

export const Button: React.FC<ButtonProps> = ({
  variant = "primary",
  loading = false,
  className,
  children,
  ...rest
}) => {
  return (
    <button
      className={clsx(
        "inline-flex items-center justify-center rounded-md px-3 py-2 text-sm font-medium transition-colors",
        "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-neutral-950",
        variantClasses[variant],
        loading && "cursor-wait opacity-80",
        className
      )}
      disabled={loading || rest.disabled}
      {...rest}
    >
      {loading && (
        <span className="mr-2 inline-block h-3 w-3 animate-spin rounded-full border-2 border-white border-t-transparent" />
      )}
      {children}
    </button>
  );
};
