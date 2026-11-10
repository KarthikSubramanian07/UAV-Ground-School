# Accessibility Conformance

UAV Ground School is committed to providing an accessible showcase site for all users, regardless of technology or ability.

## Conformance Target

This project targets **WCAG 2.2 Level AA** conformance across all showcase site pages and interactive components.

## Key Accessibility Features

- **Keyboard Navigation and Focus Management**:
  - Interactive SVG diagrams (including the Protocol Pro wiring diagram on week 4 and detection rings on week 3) are fully accessible via keyboard.
  - Arrow keys allow stepping through scored detection dots with live status announcements.
  - Wiring diagram connections and components expose clear button roles, labels, and Escape dismissal.
  - Wide tables, code blocks, terminal transcripts, and hex grids are keyboard scrollable when overflowing.
  - Visible focus indicators meet or exceed 3:1 contrast against both light and dark backgrounds.

- **Color and Contrast**:
  - Text, badges, and interface controls meet 4.5:1 minimum contrast against their backgrounds.
  - Terminal and night-mode elements use calibrated tokens for secondary text and borders.
  - States and selections are not conveyed through color alone.

- **Reflow and Responsive Layout**:
  - Navigation and header elements reflow cleanly down to 320 px viewport width without clipping or horizontal overflow.
  - Choice pills and long words wrap appropriately to prevent horizontal scrolling.

- **Screen Reader Support**:
  - Color layer buttons and interactive controls include accessible names that match visible text labels.
  - Status updates and tooltip explanations use polite live regions.

## Evaluation and Testing

Conformance was evaluated using:
- Automated audits with `axe-core` 4.14 covering WCAG 2.0, 2.1, and 2.2 AA rule sets across multiple viewport widths (1440x900, 390x844, 320x800) across all site routes.
- Manual keyboard operation, tab sequence, and focus visibility testing.
- Mobile viewport reflow testing at 320 px width.

## Feedback and Reporting

To report an accessibility barrier or suggest improvements, please submit an issue at:
https://github.com/KarthikSubramanian07/UAV-Ground-School/issues
