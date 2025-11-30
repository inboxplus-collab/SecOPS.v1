import * as React from "react";
import clsx from "clsx";

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {}

export const Card: React.FC<CardProps> = ({ className, children, ...rest }) => (
  <div
    className={clsx(
      "rounded-lg border border-neutral-800 bg-neutral-900/80 p-4 shadow-sm shadow-black/30",
      className
    )}
    {...rest}
  >
    {children}
  </div>
);
