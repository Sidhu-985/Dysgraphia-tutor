import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show SystemUiOverlayStyle;

/// Shared design tokens for DysgraphiaTutor.
///
/// Navy/cobalt interface with a blue-teal-green gradient used only on
/// decorative header regions. Writing surfaces always stay plain white so
/// handwriting and guides are easy to see.
///
/// Contrast of the main text/background pairs is checked in
/// `test/design_system_test.dart` (WCAG 2.x formula), not assumed.
/// Interface typeface (bundled, SIL OFL). Letters being taught are not set
/// in this font: they are drawn from the backend's own stroke data
/// (LetterGlyph), so they always match the model the learner traces.
const String kAppFontFamily = 'Lexend';

abstract final class AppColors {
  // Surfaces
  static const bg = Color(0xFFE6ECF5);
  static const paper = Color(0xFFFFFFFF);
  static const track = Color(0xFFD3DCEA);
  static const border = Color(0xFF7186A6);
  static const hairline = Color(0xFFC9D4E5);
  static const navy = Color(0xFF0B1F3A);
  static const white = Colors.white;

  /// Secondary text on navy (app bars, hero band).
  static const onNavyMuted = Color(0xFFC8D5EA);

  // Brand
  static const primary = Color(0xFF1D4ED8);
  static const primarySoft = Color(0xFFD9E5FF);
  static const secondary = Color(0xFF0E7490);
  static const secondarySoft = Color(0xFFD3EEF5);

  // Requested gradient palette
  static const gradientBlue = Color(0xFF50A0C8);
  static const gradientGreen = Color(0xFF2D8659);
  static const gradientSky = Color(0xFF66CCFF);
  static const gradientTeal = Color(0xFF006699);

  // Text
  static const ink = Color(0xFF0B1220);
  static const inkMuted = Color(0xFF34435C);

  // Writing
  static const guide = Color(0xFF7F93B2);
  static const startDot = Color(0xFF15803D);

  // Status
  static const success = Color(0xFF15803D);
  static const successSoft = Color(0xFFDCF3E3);
  static const errorInk = Color(0xFF991B1B);
  static const errorSoft = Color(0xFFFEE2E2);
  static const info = primary;
  static const infoSoft = primarySoft;

  // Backward-compatible aliases used by existing widgets.
  static const cream = bg;
  static const creamDeep = track;
  static const teal = secondary;
  static const tealSoft = secondarySoft;
  static const lavender = primary;
  static const lavenderSoft = primarySoft;
  static const blue = primary;
  static const blueSoft = primarySoft;
}

abstract final class AppSpacing {
  static const xs = 4.0;
  static const sm = 8.0;
  static const md = 16.0;
  static const lg = 24.0;
  static const xl = 32.0;
  static const xxl = 48.0;

  /// Minimum size of anything a child taps.
  static const touchTarget = 64.0;

  /// Page gutter: tighter on phones, roomier on tablets.
  static double gutter(double width) => width < 600 ? md : lg;
}

abstract final class AppRadii {
  static const sm = 12.0;
  static const md = 16.0;
  static const lg = 20.0;
  static const xl = 24.0;
  static const xxl = 28.0;

  static BorderRadius all(double r) => BorderRadius.circular(r);
}

abstract final class AppIconSize {
  static const sm = 20.0;
  static const md = 24.0;
  static const lg = 32.0;
  static const xl = 48.0;
}

/// Content widths that keep lines readable on wide tablets.
abstract final class AppLayout {
  static const maxContentWidth = 1180.0;
  static const maxFormWidth = 560.0;

  /// Width at which two-column layouts start.
  static const wideBreakpoint = 860.0;
}

/// Motion tokens. Every animation in the app goes through [duration], so it
/// becomes instant when the platform asks for reduced motion.
abstract final class AppMotion {
  static const fast = Duration(milliseconds: 150);
  static const medium = Duration(milliseconds: 250);
  static const slow = Duration(milliseconds: 400);
  static const curve = Curves.easeOutCubic;

  static bool reduced(BuildContext context) => MediaQuery.maybeOf(context)?.disableAnimations ?? false;

  static Duration duration(BuildContext context, Duration d) => reduced(context) ? Duration.zero : d;
}

abstract final class AppShadows {
  static List<BoxShadow> get card => [
        BoxShadow(color: AppColors.navy.withValues(alpha: 0.06), blurRadius: 10, offset: const Offset(0, 3)),
      ];
  static List<BoxShadow> get raised => [
        BoxShadow(color: AppColors.navy.withValues(alpha: 0.14), blurRadius: 22, offset: const Offset(0, 8)),
      ];
}

abstract final class AppGradients {
  /// The brand gradient. Decorative regions only, never behind writing.
  static const brand = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [
      AppColors.gradientSky,
      AppColors.gradientBlue,
      AppColors.gradientTeal,
      AppColors.gradientGreen,
    ],
    stops: [0.0, 0.35, 0.70, 1.0],
  );

  /// Navy overlay laid on top of [brand] where white text sits on it, so the
  /// text keeps at least 4.5:1 contrast across the whole gradient.
  static LinearGradient get scrim => LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [AppColors.navy.withValues(alpha: 0.72), AppColors.navy.withValues(alpha: 0.55)],
      );
}

/// Reusable gradient background. Keep writing canvases and cards white.
class AppGradientBackground extends StatelessWidget {
  final Widget child;

  /// Adds a navy scrim so white text placed directly on the gradient stays
  /// readable.
  final bool scrim;

  const AppGradientBackground({super.key, required this.child, this.scrim = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      decoration: const BoxDecoration(gradient: AppGradients.brand),
      child: scrim ? DecoratedBox(decoration: BoxDecoration(gradient: AppGradients.scrim), child: child) : child,
    );
  }
}

/// White rounded panel for handwriting or other high-contrast content.
class AppWritingSurface extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;

  const AppWritingSurface({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(AppSpacing.md),
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: padding,
      decoration: BoxDecoration(
        color: AppColors.paper,
        borderRadius: BorderRadius.circular(AppRadii.xl),
        border: Border.all(color: AppColors.border, width: 1.5),
        boxShadow: AppShadows.card,
      ),
      child: child,
    );
  }
}

ThemeData buildAppTheme() {
  final scheme = ColorScheme.fromSeed(
    seedColor: AppColors.primary,
    brightness: Brightness.light,
  ).copyWith(
    primary: AppColors.primary,
    onPrimary: Colors.white,
    primaryContainer: AppColors.primarySoft,
    onPrimaryContainer: AppColors.ink,
    secondary: AppColors.secondary,
    onSecondary: Colors.white,
    secondaryContainer: AppColors.secondarySoft,
    onSecondaryContainer: AppColors.ink,
    surface: AppColors.paper,
    onSurface: AppColors.ink,
    onSurfaceVariant: AppColors.inkMuted,
    outline: AppColors.border,
    outlineVariant: AppColors.hairline,
    surfaceTint: Colors.transparent,
    error: AppColors.errorInk,
  );

  const base = TextTheme(
    // Lexend runs wider than Roboto, so sizes are ~1 pt smaller for the same
    // optical size. Body text never goes below 16 (15 for adult-only notes).
    displaySmall: TextStyle(fontSize: 32, fontWeight: FontWeight.w700, height: 1.15, letterSpacing: -0.8),
    headlineMedium: TextStyle(fontSize: 27, fontWeight: FontWeight.w700, height: 1.2, letterSpacing: -0.5),
    headlineSmall: TextStyle(fontSize: 23, fontWeight: FontWeight.w600, height: 1.25, letterSpacing: -0.3),
    titleLarge: TextStyle(fontSize: 20, fontWeight: FontWeight.w600, height: 1.3, letterSpacing: -0.2),
    titleMedium: TextStyle(fontSize: 18, fontWeight: FontWeight.w600, height: 1.3),
    titleSmall: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, height: 1.3),
    bodyLarge: TextStyle(fontSize: 18, height: 1.5),
    bodyMedium: TextStyle(fontSize: 16, height: 1.5),
    bodySmall: TextStyle(fontSize: 15, height: 1.45),
    labelLarge: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
    labelMedium: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, letterSpacing: 0.6),
  );
  final textTheme = base.apply(
    bodyColor: AppColors.ink,
    displayColor: AppColors.ink,
  );

  return ThemeData(
    useMaterial3: true,
    fontFamily: kAppFontFamily,
    // One transition everywhere (Android, iOS, web, desktop) so moving between
    // screens always feels the same.
    pageTransitionsTheme: PageTransitionsTheme(builders: {
      for (final p in TargetPlatform.values) p: const ZoomPageTransitionsBuilder(),
    }),
    focusColor: AppColors.primary.withValues(alpha: 0.18),
    hoverColor: AppColors.primary.withValues(alpha: 0.06),
    colorScheme: scheme,
    scaffoldBackgroundColor: AppColors.bg,
    textTheme: textTheme,
    visualDensity: VisualDensity.standard,
    appBarTheme: const AppBarTheme(
      backgroundColor: AppColors.navy,
      foregroundColor: Colors.white,
      surfaceTintColor: Colors.transparent,
      scrolledUnderElevation: 0,
      elevation: 0,
      centerTitle: false,
      toolbarHeight: 64,
      systemOverlayStyle: SystemUiOverlayStyle.light,
      titleTextStyle: TextStyle(
        fontFamily: kAppFontFamily,
        fontSize: 21,
        fontWeight: FontWeight.w600,
        color: Colors.white,
        letterSpacing: -0.2,
      ),
      iconTheme: IconThemeData(color: Colors.white),
      actionsIconTheme: IconThemeData(color: Colors.white),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.primary,
        foregroundColor: Colors.white,
        disabledBackgroundColor: AppColors.track,
        disabledForegroundColor: AppColors.inkMuted,
        elevation: 0,
        minimumSize: const Size(AppSpacing.touchTarget * 2, AppSpacing.touchTarget),
        padding: const EdgeInsets.symmetric(horizontal: AppSpacing.lg, vertical: AppSpacing.md),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        textStyle: const TextStyle(fontFamily: kAppFontFamily, fontSize: 19, fontWeight: FontWeight.w600),
      ),
    ),
    // Dark text on a cobalt outline. Controls on navy (e.g. the letter-case
    // switch in the letter page's app bar) override this locally; the global
    // style stays dark so every other outlined button is unaffected.
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        foregroundColor: AppColors.ink,
        backgroundColor: AppColors.paper,
        minimumSize: const Size(AppSpacing.touchTarget, AppSpacing.touchTarget),
        padding: const EdgeInsets.symmetric(horizontal: AppSpacing.md, vertical: AppSpacing.md),
        // Quiet secondary style: a neutral outline (3.5:1 against the page)
        // so secondary buttons never compete with the cobalt primary action.
        side: const BorderSide(color: AppColors.border, width: 1.5),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        textStyle: const TextStyle(fontFamily: kAppFontFamily, fontSize: 17, fontWeight: FontWeight.w600),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        foregroundColor: AppColors.primary,
        minimumSize: const Size(AppSpacing.touchTarget, 48),
        padding: const EdgeInsets.symmetric(horizontal: AppSpacing.md, vertical: AppSpacing.sm),
        textStyle: const TextStyle(fontFamily: kAppFontFamily, fontSize: 18, fontWeight: FontWeight.w600),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      ),
    ),
    chipTheme: ChipThemeData(
      backgroundColor: AppColors.paper,
      selectedColor: AppColors.primarySoft,
      side: const BorderSide(color: AppColors.hairline, width: 1.5),
      labelStyle:
          const TextStyle(fontFamily: kAppFontFamily, fontSize: 16, fontWeight: FontWeight.w600, color: AppColors.ink),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      padding: const EdgeInsets.symmetric(horizontal: AppSpacing.sm, vertical: AppSpacing.xs),
    ),
    progressIndicatorTheme: const ProgressIndicatorThemeData(
      color: AppColors.primary,
      linearTrackColor: AppColors.track,
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: AppColors.paper,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(AppRadii.xl)),
      titleTextStyle:
          const TextStyle(fontFamily: kAppFontFamily, fontSize: 24, fontWeight: FontWeight.w700, color: AppColors.ink),
      contentTextStyle: const TextStyle(fontFamily: kAppFontFamily, fontSize: 18, height: 1.45, color: AppColors.ink),
    ),
    tooltipTheme: const TooltipThemeData(
      textStyle: TextStyle(fontFamily: kAppFontFamily, fontSize: 15, color: Colors.white),
      decoration: BoxDecoration(color: AppColors.navy, borderRadius: BorderRadius.all(Radius.circular(8))),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: AppColors.paper,
      labelStyle: const TextStyle(fontFamily: kAppFontFamily, fontSize: 18, color: AppColors.inkMuted),
      helperStyle: const TextStyle(fontFamily: kAppFontFamily, fontSize: 15, color: AppColors.inkMuted),
      helperMaxLines: 2,
      errorMaxLines: 3,
      errorStyle: const TextStyle(fontFamily: kAppFontFamily, fontSize: 15, color: AppColors.errorInk),
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(16)),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(16),
        borderSide: const BorderSide(color: AppColors.border, width: 2),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(16),
        borderSide: const BorderSide(color: AppColors.primary, width: 2.5),
      ),
    ),
    cardTheme: CardThemeData(
      color: AppColors.paper,
      surfaceTintColor: Colors.transparent,
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
    ),
    dividerTheme: const DividerThemeData(color: AppColors.hairline, thickness: 1, space: 1),
  );
}
