import * as React from "react";
import clsx from "clsx";

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  helperText?: string;
}

export const Input: React.FC<InputProps> = ({
  label,
  helperText,
  className,
  id,
  ...rest
}) => {
  const inputId = id || rest.name || undefined;
  return (
    <div className="flex flex-col gap-1">
      {label && (
        <label
          htmlFor={inputId}
          className="text-xs font-medium uppercase tracking-wide text-neutral-300"
        >
          {label}
        </label>
      )}
      <input
        id={inputId}
        className={clsx(
          "w-full rounded-md border border-neutral-700 bg-neutral-950 px-3 py-2 text-sm text-neutral-50",
          "placeholder:text-neutral-500 focus:outline-none focus:ring-2 focus:ring-blue-500",
          className
        )}
        {...rest}
      />
      {helperText && (
        <p className="text-xs text-neutral-500">{helperText}</p>
      )}
    </div>
  );
};
