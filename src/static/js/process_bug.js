$(document).ready(function() {
    $("form").on('submit', function() {
        $.ajax({url: "/model",
                data: {'product': $("input[name='product']").val().toLowerCase(),
                        'component': $("input[name='component']").val().toLowerCase(),
                        'assignee': $("input[name='assignee']").val().toLowerCase(),
                        'cc': $("input[name='cc']").val().toLowerCase(),
                        'short_desc': $("input[name='short_desc']").val().toLowerCase(),
                        'op_sys': $("input[name='op_sys']").val().toLowerCase(),
                        'desc': $("textarea[name='desc']").val().toLowerCase()
                },
                async: false,
                success: function(result) {
                    $("span#model_result").html(result)
                }})
    })
});
