import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/alphabet.dart';
import '../models/exercise.dart';
import '../models/exercise_instruction.dart';
import '../models/session_summary.dart';
import '../services/api_service.dart';
import '../theme/app_theme.dart';
import '../widgets/alphabet_section.dart';
import '../widgets/app_components.dart';
import '../widgets/letter_glyph.dart';
import '../widgets/policy_status_card.dart';
import '../widgets/shape_preview.dart';
import 'letter_detail_screen.dart';
import 'writing_screen.dart';

/// Route name of the dashboard, so other screens can return to it.
const String kExercisesRouteName = '/exercises';

/// Dashboard of the exercises served by the backend. Letters are shown in the
/// "Learn & Write the Alphabet" section (lowercase and uppercase tabs, A-Z);
/// picking a letter opens its learning page. Words and shapes are listed
/// below; picking one starts a backend session at that exercise. The
/// supervising adult sees which policy the backend is actually running.
class ExerciseSelectionScreen extends StatefulWidget {
  final ApiService api;
  final String childId;
  final PracticeLog? practiceLog;

  const ExerciseSelectionScreen({super.key, required this.api, required this.childId, this.practiceLog});

  @override
  State<ExerciseSelectionScreen> createState() => _ExerciseSelectionScreenState();
}

class _ExerciseSelectionScreenState extends State<ExerciseSelectionScreen> {
  late Future<List<Exercise>> _exercises;
  late final PracticeLog _log;
  BackendHealth? _health;
  String? _healthError;
  String? _startingId;
  String? _error;
  String? _selectedLetterId;

  bool get _starting => _startingId != null;

  @override
  void initState() {
    super.initState();
    _log = widget.practiceLog ?? PracticeLog();
    _log.addListener(_onLog);
    _load();
  }

  @override
  void dispose() {
    _log.removeListener(_onLog);
    super.dispose();
  }

  void _onLog() {
    if (mounted) setState(() {});
  }

  void _load() {
    _exercises = widget.api.listExercises();
    _healthError = null;
    _error = null;
    widget.api.health().then((h) {
      if (mounted) setState(() => _health = h);
    }).catchError((Object e) {
      if (mounted) setState(() => _healthError = describeError(e));
    });
  }

  Future<void> _openLetter(AlphabetIndex index, Exercise ex, List<String> sequence) async {
    setState(() => _selectedLetterId = ex.exerciseId);
    await Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => LetterDetailScreen(
        api: widget.api,
        childId: widget.childId,
        index: index,
        initial: ex,
        sequence: sequence,
        log: _log,
      ),
    ));
  }

  /// The suggested next letter: the first small letter in a-z order with no
  /// accepted attempt in this run of the app (PracticeLog). When all are
  /// practised it suggests "a" again and says so. Null without letters.
  (Exercise, List<String>, bool)? _nextLetter(AlphabetIndex index) {
    final letters = [
      for (final l in index.letters(LetterCase.lower))
        if (l.$2 != null) l
    ];
    if (letters.isEmpty) return null;
    final seq = [for (final l in letters) l.$1];
    for (final l in letters) {
      if (!_log.practised(l.$2!.exerciseId)) return (l.$2!, seq, false);
    }
    return (letters.first.$2!, seq, true);
  }

  Future<void> _pick(Exercise ex) async {
    setState(() {
      _startingId = ex.exerciseId;
      _error = null;
    });
    try {
      final resp = await widget.api.startSession(widget.childId, exerciseId: ex.exerciseId);
      if (!mounted) return;
      final exercise = Exercise.fromJson(resp['exercise'] as Map<String, dynamic>);
      // push (not pushReplacement): the dashboard stays underneath, so the
      // session summary can offer "Choose another exercise".
      await Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => WritingScreen(
          api: widget.api,
          sessionId: resp['session_id'] as String,
          childId: widget.childId,
          initialExercise: exercise,
          onSessionEnded: () {},
        ),
      ));
    } catch (e) {
      if (mounted) setState(() => _error = describeError(e));
    } finally {
      if (mounted) setState(() => _startingId = null);
    }
  }

  @override
  Widget build(BuildContext context) {
    final narrow = MediaQuery.sizeOf(context).width < 600;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Choose an exercise'),
        actions: [
          // Phones: the learner code moves into the header so the title fits.
          if (!narrow)
            Padding(
              padding: const EdgeInsets.only(right: AppSpacing.md),
              child: Center(
                child: Semantics(
                  label: 'Learner code ${widget.childId}',
                  child: ExcludeSemantics(
                    child: ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 180),
                      child: AppTag(
                        label: widget.childId,
                        icon: Icons.badge_outlined,
                        background: Colors.white.withValues(alpha: 0.14),
                        foreground: Colors.white,
                      ),
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
      body: LayoutBuilder(builder: (context, c) {
        final gutter = AppSpacing.gutter(c.maxWidth);
        final side = math.max(gutter, (c.maxWidth - AppLayout.maxContentWidth) / 2);
        EdgeInsets pad({double top = 0, double bottom = 0}) => EdgeInsets.fromLTRB(side, top, side, bottom);
        return FutureBuilder<List<Exercise>>(
          future: _exercises,
          builder: (context, snap) {
            final alphabet = snap.hasData ? AlphabetIndex.fromExercises(snap.data!) : null;
            final next = alphabet == null ? null : _nextLetter(alphabet);
            final slivers = <Widget>[
              SliverToBoxAdapter(
                  child: _Header(
                      childId: c.maxWidth < 600 ? widget.childId : null,
                      compact: c.maxWidth < 600,
                      wide: c.maxWidth >= AppLayout.wideBreakpoint,
                      nextUp: next == null
                          ? null
                          : _NextUpCard(
                              exercise: next.$1,
                              allDone: next.$3,
                              enabled: !_starting,
                              onTap: () => _openLetter(alphabet!, next.$1, next.$2),
                            ),
                      padding: pad(top: c.maxWidth < 600 ? AppSpacing.md : AppSpacing.xl, bottom: AppSpacing.xl))),
              SliverPadding(
                padding: pad(top: AppSpacing.md),
                sliver: SliverToBoxAdapter(child: PolicyStatusCard(health: _health, error: _healthError)),
              ),
              if (_error != null)
                SliverPadding(
                  padding: pad(top: AppSpacing.md),
                  sliver: SliverToBoxAdapter(
                    child: AppStatusBanner.error(
                      title: 'Could not start the exercise',
                      message: _error!,
                      actionLabel: 'Try again',
                      onAction: () => setState(_load),
                    ),
                  ),
                ),
            ];
            if (snap.connectionState != ConnectionState.done) {
              slivers.add(const SliverFillRemaining(
                hasScrollBody: false,
                child: AppLoadingState(message: 'Loading exercises...'),
              ));
            } else if (snap.hasError) {
              slivers.add(SliverPadding(
                padding: pad(top: AppSpacing.md, bottom: AppSpacing.xl),
                sliver: SliverToBoxAdapter(
                  child: AppStatusBanner.error(
                    title: 'Exercises could not be loaded',
                    message: describeError(snap.error!),
                    actionLabel: 'Try again',
                    icon: Icons.wifi_off_rounded,
                    onAction: () => setState(_load),
                  ),
                ),
              ));
            } else {
              slivers.addAll(_content(snap.data!, pad));
            }
            return CustomScrollView(slivers: slivers);
          },
        );
      }),
    );
  }

  List<Widget> _content(List<Exercise> items, EdgeInsets Function({double top, double bottom}) pad) {
    if (items.isEmpty) {
      return [
        const SliverFillRemaining(
          hasScrollBody: false,
          child: AppEmptyState(
            icon: Icons.inbox_outlined,
            title: 'No exercises yet',
            message: 'The tutor server sent an empty exercise list. Check the server, then try again.',
          ),
        ),
      ];
    }
    final alphabet = AlphabetIndex.fromExercises(items);
    final hasAlphabet = alphabet.count(LetterCase.lower) + alphabet.count(LetterCase.upper) > 0;
    // Letters the alphabet section cannot place (malformed records) stay listed below.
    final sections = [
      if (!hasAlphabet) (ExerciseCategory.letter, 'Letters', 'Practise single letters.', Icons.abc_rounded),
      (ExerciseCategory.word, 'Words', 'Short words that join several letters.', Icons.short_text_rounded),
      (ExerciseCategory.shape, 'Shapes', 'Lines, curves and loops that warm up the hand.', Icons.gesture_rounded),
      (ExerciseCategory.unknown, 'More practice', null, Icons.more_horiz_rounded),
    ];
    return [
      if (hasAlphabet)
        SliverPadding(
          padding: pad(top: AppSpacing.lg),
          sliver: SliverToBoxAdapter(
            child: AlphabetSection(
              index: alphabet,
              log: _log,
              selectedExerciseId: _selectedLetterId,
              enabled: !_starting,
              onOpen: (ex, seq) => _openLetter(alphabet, ex, seq),
            ),
          ),
        ),
      for (final (category, title, subtitle, icon) in sections)
        if (items.any((e) => e.category == category)) ...[
          SliverPadding(
            padding: pad(top: AppSpacing.xl, bottom: AppSpacing.md),
            sliver: SliverToBoxAdapter(child: AppSectionHeader(title: title, subtitle: subtitle, icon: icon)),
          ),
          SliverPadding(
            padding: pad(),
            sliver: SliverGrid(
              gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
                maxCrossAxisExtent: 240,
                mainAxisSpacing: AppSpacing.md,
                crossAxisSpacing: AppSpacing.md,
                childAspectRatio: 0.92,
              ),
              delegate: SliverChildListDelegate([
                for (final ex in items.where((e) => e.category == category))
                  ExerciseCard(
                    exercise: ex,
                    enabled: !_starting,
                    starting: _startingId == ex.exerciseId,
                    onTap: () => _pick(ex),
                  ),
              ]),
            ),
          ),
        ],
      const SliverToBoxAdapter(child: SizedBox(height: AppSpacing.xxl)),
    ];
  }
}

/// Greeting band on the signature hero surface, with the suggested next
/// letter as the one obvious action. White text on navy (4.5:1+).
class _Header extends StatelessWidget {
  final EdgeInsets padding;
  final bool compact;
  final bool wide;
  final Widget? nextUp;

  /// Shown in the header on phones, where the app bar has no room for it.
  final String? childId;

  const _Header({required this.padding, this.compact = false, this.wide = false, this.childId, this.nextUp});

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final greeting = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        if (childId != null) ...[
          Semantics(
            label: 'Learner code $childId',
            child: ExcludeSemantics(
              child: AppTag(
                label: childId!,
                icon: Icons.badge_outlined,
                background: Colors.white.withValues(alpha: 0.12),
                foreground: Colors.white,
              ),
            ),
          ),
          const SizedBox(height: AppSpacing.md),
        ],
        Semantics(
          header: true,
          child: Text('What would you like to practise?',
              style: (compact ? text.headlineSmall : text.headlineMedium)?.copyWith(color: Colors.white)),
        ),
        const SizedBox(height: AppSpacing.sm),
        Text(
          compact
              ? 'Start with a letter. Words and shapes are further down.'
              : 'Start with a letter: watch how it is written, trace it, then write it on your own. '
                  'Words and shapes are further down.',
          style: text.bodyLarge?.copyWith(color: AppColors.onNavyMuted),
        ),
      ],
    );
    return AppHeroBand(
      child: Padding(
        padding: padding,
        child: wide && nextUp != null
            ? Row(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Expanded(flex: 3, child: greeting),
                  const SizedBox(width: AppSpacing.xl),
                  Expanded(flex: 2, child: nextUp!),
                ],
              )
            : Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  greeting,
                  if (nextUp != null) ...[const SizedBox(height: AppSpacing.lg), nextUp!]
                ],
              ),
      ),
    );
  }
}

/// "Next letter" call to action: the suggested letter, drawn from its own
/// stroke data on writing lines, and one button that opens it.
class _NextUpCard extends StatelessWidget {
  final Exercise exercise;
  final bool allDone;
  final bool enabled;
  final VoidCallback onTap;

  const _NextUpCard({required this.exercise, required this.allDone, required this.enabled, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final name = letterName(exercise.target!);
    return Semantics(
      button: true,
      enabled: enabled,
      label: allDone
          ? 'All small letters practised today. Start again with $name'
          : 'Next letter to learn: $name. Open it',
      child: ExcludeSemantics(
        child: AppPressable(
          enabled: enabled,
          child: Material(
            key: const Key('next-up'),
            color: AppColors.paper,
            borderRadius: BorderRadius.circular(AppRadii.xl),
            clipBehavior: Clip.antiAlias,
            elevation: 0,
            child: InkWell(
              onTap: enabled ? onTap : null,
              child: Padding(
                padding: const EdgeInsets.all(AppSpacing.md),
                child: Row(
                  children: [
                    Container(
                      width: 84,
                      height: 84,
                      padding: const EdgeInsets.all(8),
                      decoration:
                          BoxDecoration(color: AppColors.primarySoft, borderRadius: BorderRadius.circular(AppRadii.md)),
                      child: LetterGlyph.of(exercise, showLines: true),
                    ),
                    const SizedBox(width: AppSpacing.md),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          AppEyebrow(allDone ? 'All 26 practised today' : 'Next letter in a–z',
                              color: AppColors.secondary),
                          const SizedBox(height: 2),
                          Text(allDone ? 'Start again with $name' : 'Learn $name',
                              style: text.titleLarge, maxLines: 2, overflow: TextOverflow.ellipsis),
                        ],
                      ),
                    ),
                    const SizedBox(width: AppSpacing.sm),
                    Container(
                      width: 48,
                      height: 48,
                      decoration: const BoxDecoration(color: AppColors.primary, shape: BoxShape.circle),
                      child: const Icon(Icons.arrow_forward_rounded, color: Colors.white),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// A word or shape exercise. Tapping it starts a backend session at once, so
/// the card shows a "Starting..." state while that request runs.
class ExerciseCard extends StatelessWidget {
  final Exercise exercise;
  final bool enabled;
  final bool starting;
  final VoidCallback onTap;

  const ExerciseCard({
    super.key,
    required this.exercise,
    required this.enabled,
    required this.onTap,
    this.starting = false,
  });

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final label = exerciseLabel(exercise);
    final title = label.isEmpty ? exercise.prompt : label; // unknown metadata: the backend's own wording
    return Semantics(
        button: true,
        enabled: enabled,
        label: '$title, level ${exercise.difficulty}${starting ? ', starting' : ''}',
        child: ExcludeSemantics(
          child: AppPressable(
            enabled: enabled,
            child: Material(
              key: Key('exercise-card-${exercise.exerciseId}'),
              color: AppColors.paper,
              borderRadius: BorderRadius.circular(AppRadii.xl),
              clipBehavior: Clip.antiAlias,
              child: InkWell(
                onTap: enabled ? onTap : null,
                child: Ink(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(AppRadii.xl),
                    border: Border.all(
                        color: starting ? AppColors.primary : AppColors.hairline, width: starting ? 2.5 : 1.5),
                  ),
                  padding: const EdgeInsets.all(AppSpacing.sm + 4),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Expanded(
                        child: Container(
                          decoration: BoxDecoration(
                              color: AppColors.bg.withValues(alpha: 0.6), borderRadius: BorderRadius.circular(16)),
                          padding: const EdgeInsets.all(AppSpacing.sm),
                          child: starting
                              ? const Center(child: CircularProgressIndicator())
                              : ShapePreview.strokes(
                                  strokes: exercise.guideStrokes,
                                  ascender: exercise.writingLines?.ascender,
                                  baseline: exercise.writingLines?.baseline,
                                  descender: exercise.writingLines?.descender,
                                ),
                        ),
                      ),
                      const SizedBox(height: AppSpacing.sm),
                      Text(title, style: text.titleMedium, maxLines: 1, overflow: TextOverflow.ellipsis),
                      const SizedBox(height: 2),
                      Row(
                        children: [
                          Expanded(
                            child: Text(starting ? 'Starting...' : 'Level ${exercise.difficulty}',
                                style: text.bodyMedium?.copyWith(color: AppColors.inkMuted)),
                          ),
                          const Icon(Icons.arrow_forward_rounded, color: AppColors.primary),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ));
  }
}
