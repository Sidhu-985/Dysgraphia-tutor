import 'package:flutter/material.dart';

import '../models/alphabet.dart';
import '../models/exercise.dart';
import '../theme/app_theme.dart';
import 'app_components.dart';
import 'letter_glyph.dart';

/// "Learn & Write the Alphabet": two tabs (lowercase a-z, uppercase A-Z),
/// each a responsive grid of 26 letter tiles in alphabetical order, with an
/// optional practice-group filter. Tapping a tile opens that letter; nothing
/// forces the child through the whole alphabet.
class AlphabetSection extends StatefulWidget {
  final AlphabetIndex index;
  final PracticeLog log;
  final String? selectedExerciseId;
  final LetterCase initialCase;

  /// Called with the chosen letter and the practice sequence it was chosen
  /// from (all 26 letters of that case, or the letters of the chosen group).
  final void Function(Exercise exercise, List<String> sequence) onOpen;
  final bool enabled;

  const AlphabetSection({
    super.key,
    required this.index,
    required this.log,
    required this.onOpen,
    this.selectedExerciseId,
    this.initialCase = LetterCase.lower,
    this.enabled = true,
  });

  @override
  State<AlphabetSection> createState() => _AlphabetSectionState();
}

class _AlphabetSectionState extends State<AlphabetSection> with SingleTickerProviderStateMixin {
  late final TabController _tabs;
  PracticeGroup? _group;

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 2, vsync: this, initialIndex: widget.initialCase == LetterCase.lower ? 0 : 1);
    _tabs.addListener(() {
      if (!_tabs.indexIsChanging && mounted) setState(() {});
    });
  }

  @override
  void dispose() {
    _tabs.dispose();
    super.dispose();
  }

  LetterCase get _case => _tabs.index == 0 ? LetterCase.lower : LetterCase.upper;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final c = _case;
    final all = widget.index.letters(c);
    final shown = _group == null
        ? all
        : [
            for (final l in all)
              if (_group!.lettersFor(c).contains(l.$1)) l
          ];
    final sequence = [
      for (final l in shown)
        if (l.$2 != null) l.$1
    ];
    final ids = [
      for (final l in all)
        if (l.$2 != null) l.$2!.exerciseId
    ];
    final practised = widget.log.practisedCount(ids);
    final narrow = MediaQuery.sizeOf(context).width < 600;

    return Container(
      key: const Key('alphabet-section'),
      padding: const EdgeInsets.all(AppSpacing.md),
      decoration: BoxDecoration(
        color: AppColors.paper,
        borderRadius: BorderRadius.circular(AppRadii.xxl),
        border: Border.all(color: AppColors.hairline, width: 1.5),
        boxShadow: AppShadows.card,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Semantics(
            header: true,
            child: Row(
              children: [
                ExcludeSemantics(
                  child: Container(
                    width: 56,
                    height: 56,
                    decoration:
                        BoxDecoration(color: AppColors.primarySoft, borderRadius: BorderRadius.circular(AppRadii.md)),
                    child: const Center(
                        child: Text('Aa',
                            style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800, color: AppColors.primary))),
                  ),
                ),
                const SizedBox(width: AppSpacing.md),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Learn & Write the Alphabet', style: text.headlineSmall),
                      Text('Pick a letter to see how it is written, then trace it or write it yourself.',
                          style: text.bodyMedium?.copyWith(color: AppColors.inkMuted)),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: AppSpacing.md),
          Container(
            padding: const EdgeInsets.all(4),
            decoration: BoxDecoration(color: AppColors.bg, borderRadius: BorderRadius.circular(20)),
            child: TabBar(
              controller: _tabs,
              indicatorSize: TabBarIndicatorSize.tab,
              dividerColor: Colors.transparent,
              splashBorderRadius: BorderRadius.circular(16),
              indicator: BoxDecoration(
                color: AppColors.navy,
                borderRadius: BorderRadius.circular(16),
                boxShadow: AppShadows.card,
              ),
              labelColor: Colors.white,
              unselectedLabelColor: AppColors.ink,
              labelStyle: const TextStyle(fontSize: 19, fontWeight: FontWeight.w700),
              unselectedLabelStyle: const TextStyle(fontSize: 19, fontWeight: FontWeight.w600),
              labelPadding: const EdgeInsets.symmetric(horizontal: AppSpacing.sm),
              // Phones get the short form so neither label is cut off.
              tabs: [
                Tab(key: const Key('tab-lowercase'), height: 56, text: narrow ? 'Small a–z' : 'Lowercase a–z'),
                Tab(key: const Key('tab-uppercase'), height: 56, text: narrow ? 'Capital A–Z' : 'Uppercase A–Z'),
              ],
            ),
          ),
          const SizedBox(height: AppSpacing.md),
          _ProgressRow(
              practised: practised,
              total: ids.length,
              caseLabel: c == LetterCase.lower ? 'small' : 'capital',
              short: narrow),
          const SizedBox(height: AppSpacing.md),
          LayoutBuilder(builder: (context, constraints) {
            final chips = [
              _GroupChip(label: 'All letters', selected: _group == null, onTap: () => setState(() => _group = null)),
              for (final g in kPracticeGroups)
                if (g.lettersFor(c).isNotEmpty)
                  _GroupChip(label: g.title, selected: _group?.id == g.id, onTap: () => setState(() => _group = g)),
            ];
            // Phones: one scrollable row keeps the grid near the top.
            // Tablets: every group stays visible.
            if (constraints.maxWidth < 600) {
              return SingleChildScrollView(scrollDirection: Axis.horizontal, child: Row(children: chips));
            }
            return Wrap(runSpacing: AppSpacing.xs, children: chips);
          }),
          const SizedBox(height: AppSpacing.md),
          LayoutBuilder(builder: (context, constraints) {
            // 4 columns on a narrow phone, up to 9 on a wide tablet. A Wrap
            // (not a nested GridView) so the dashboard has one scrollable and
            // "scroll to this letter" works.
            final cols = (constraints.maxWidth / 112).floor().clamp(4, 9).toInt();
            const gap = AppSpacing.sm;
            final w = ((constraints.maxWidth - gap * (cols - 1)) / cols).floorToDouble();
            return Wrap(
              key: Key('alphabet-grid-${c.name}'),
              spacing: gap,
              runSpacing: gap,
              children: [
                for (final (ch, ex) in shown)
                  SizedBox(
                    width: w,
                    height: w / 0.86,
                    child: LetterTile(
                      char: ch,
                      exercise: ex,
                      selected: ex != null && ex.exerciseId == widget.selectedExerciseId,
                      practised: ex != null && widget.log.practised(ex.exerciseId),
                      enabled: widget.enabled && ex != null,
                      onTap: ex == null ? null : () => widget.onOpen(ex, sequence),
                    ),
                  ),
              ],
            );
          }),
        ],
      ),
    );
  }
}

class _ProgressRow extends StatelessWidget {
  final int practised;
  final int total;
  final String caseLabel;
  final bool short;

  const _ProgressRow({required this.practised, required this.total, required this.caseLabel, this.short = false});

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final value = total == 0 ? 0.0 : practised / total;
    return Semantics(
      label: 'Practised $practised of $total $caseLabel letters today',
      child: ExcludeSemantics(
        child: Row(
          children: [
            const Icon(Icons.check_circle_rounded, color: AppColors.success, size: AppIconSize.sm),
            const SizedBox(width: AppSpacing.sm),
            Expanded(
              flex: 3,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: TweenAnimationBuilder<double>(
                  tween: Tween(end: value),
                  duration: AppMotion.duration(context, AppMotion.slow),
                  curve: AppMotion.curve,
                  builder: (_, v, __) => LinearProgressIndicator(
                      value: v, minHeight: 10, backgroundColor: AppColors.track, color: AppColors.success),
                ),
              ),
            ),
            const SizedBox(width: AppSpacing.sm),
            Flexible(
              flex: 2,
              child: Text(short ? '$practised / $total done' : '$practised / $total practised today',
                  style: text.bodyMedium?.copyWith(color: AppColors.inkMuted), maxLines: 2),
            ),
          ],
        ),
      ),
    );
  }
}

class _GroupChip extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;

  const _GroupChip({required this.label, required this.selected, required this.onTap});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(right: AppSpacing.sm),
        child: ChoiceChip(
          label: Text(label,
              style: TextStyle(
                  fontSize: 17, fontWeight: selected ? FontWeight.w700 : FontWeight.w500, color: AppColors.ink)),
          selected: selected,
          onSelected: (_) => onTap(),
          showCheckmark: true,
          checkmarkColor: AppColors.primary,
          selectedColor: AppColors.primarySoft,
          backgroundColor: AppColors.paper,
          side: BorderSide(color: selected ? AppColors.primary : AppColors.hairline, width: 1.5),
          materialTapTargetSize: MaterialTapTargetSize.padded,
        ),
      );
}

/// One letter tile: the letter large, its name, a practised badge (only when
/// this app run has a submitted attempt for it) and a clear selected state.
/// State is shown with icons and borders, never colour alone.
class LetterTile extends StatelessWidget {
  final String char;
  final Exercise? exercise;
  final bool selected;
  final bool practised;
  final bool enabled;
  final VoidCallback? onTap;

  const LetterTile({
    super.key,
    required this.char,
    required this.exercise,
    required this.selected,
    required this.practised,
    required this.enabled,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final ex = exercise;
    final zone = ex == null ? '' : zoneDescription(ex.letterZone);
    final label = [
      letterName(char),
      if (zone.isNotEmpty) zone,
      if (practised) 'practised',
      if (selected) 'selected',
      if (ex == null) 'not available on this server',
    ].join(', ');
    final unavailable = ex == null;
    final borderColor = selected ? AppColors.secondary : (practised ? AppColors.success : AppColors.hairline);
    return Semantics(
      button: true,
      enabled: enabled,
      selected: selected,
      label: label,
      child: ExcludeSemantics(
        child: AppPressable(
          enabled: enabled,
          child: Material(
            key: Key('letter-tile-$char${char.toUpperCase() == char ? '-upper' : '-lower'}'),
            color: selected ? AppColors.secondarySoft : (unavailable ? AppColors.track : Colors.white),
            borderRadius: BorderRadius.circular(AppRadii.lg),
            clipBehavior: Clip.antiAlias,
            child: InkWell(
              onTap: enabled ? onTap : null,
              splashColor: AppColors.secondary.withValues(alpha: 0.16),
              highlightColor: AppColors.secondary.withValues(alpha: 0.08),
              child: AnimatedContainer(
                duration: AppMotion.duration(context, AppMotion.medium),
                curve: AppMotion.curve,
                constraints: const BoxConstraints(minHeight: AppSpacing.touchTarget, minWidth: AppSpacing.touchTarget),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(AppRadii.lg),
                  border: Border.all(color: borderColor, width: selected ? 3 : 1.5),
                ),
                child: Stack(
                  children: [
                    Positioned.fill(
                      child: Padding(
                        padding: const EdgeInsets.fromLTRB(10, 10, 10, 10),
                        child: Column(
                          children: [
                            // The letter as the tutor teaches it, drawn from the
                            // backend's strokes on the shared four-line frame.
                            Expanded(
                              child: unavailable
                                  ? FittedBox(
                                      child: Text(char,
                                          style: const TextStyle(
                                              fontSize: 64, fontWeight: FontWeight.w500, color: AppColors.inkMuted)))
                                  : LetterGlyph.of(ex,
                                      showLines: true, color: selected ? AppColors.navy : AppColors.ink),
                            ),
                            if (unavailable)
                              const FittedBox(
                                child: Text('not available',
                                    style: TextStyle(
                                        fontSize: 13, color: AppColors.inkMuted, fontWeight: FontWeight.w600)),
                              ),
                          ],
                        ),
                      ),
                    ),
                    if (practised)
                      const Positioned(
                        top: 5,
                        right: 5,
                        child: Icon(Icons.check_circle_rounded, size: 22, color: AppColors.success),
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
